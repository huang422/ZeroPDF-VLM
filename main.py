#!/usr/bin/env python3
"""Main program: Process all files in input/ directory with nested structure (date/case_id/*.pdf)."""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

from vlm_pdf_recognizer.pipeline import DocumentProcessor, ProcessingResult
from vlm_pdf_recognizer.output import (
    save_result, save_batch_summary, save_batch_summary_with_vlm,
    save_vlm_visualization, save_preprocessed_rois
)
from vlm_pdf_recognizer.recognition.vlm_recognizer import DocumentRecognitionOutput


def scan_nested_input(input_dir: str):
    """Scan nested input directory structure: input/date/case_id/*.pdf

    Args:
        input_dir: Root input directory path

    Returns:
        List of tuples: (file_path, date_dir, case_id)
        Sorted by date_dir, case_id, then filename
    """
    input_path = Path(input_dir)
    supported_exts = {'.jpg', '.jpeg', '.png', '.pdf'}
    file_entries = []

    # Walk through nested structure: date/case_id/files
    for date_dir in sorted(input_path.iterdir()):
        if not date_dir.is_dir():
            continue

        for case_dir in sorted(date_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            case_id = case_dir.name
            date_name = date_dir.name

            for file_path in sorted(case_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                    file_entries.append((file_path, date_name, case_id))

    return file_entries


def main():
    """Process all files in input/ directory with nested structure and save to output/."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="VLM PDF Recognizer - Batch Processing")
    parser.add_argument(
        '--disable-vlm',
        action='store_true',
        help='Disable VLM-based ROI content recognition (only extract ROIs without VLM inference)'
    )
    args = parser.parse_args()

    # VLM is enabled by default (unless --disable-vlm is specified)
    args.enable_vlm = not args.disable_vlm

    print("=" * 70)
    print("VLM PDF Recognizer - Batch Processing (Ollama glm-ocr)")
    if args.enable_vlm:
        print("VLM Recognition: ENABLED (default)")
    else:
        print("VLM Recognition: DISABLED (--disable-vlm)")
    print("=" * 70)
    print()

    # Configuration
    input_dir = "input"
    output_dir = "output"
    templates_dir = "data"

    # Check input directory exists
    if not os.path.exists(input_dir):
        print(f"   Error: Input directory '{input_dir}/' not found!")
        print(f"   Please create it and add your documents.")
        sys.exit(1)

    # Scan nested input structure: input/date/case_id/*.pdf
    file_entries = scan_nested_input(input_dir)

    if not file_entries:
        print(f"   No files found in '{input_dir}/' directory!")
        print(f"   Expected structure: {input_dir}/date/case_id/*.pdf")
        print(f"   Supported formats: .jpg, .jpeg, .png, .pdf")
        sys.exit(1)

    # Group by case_id for display
    cases = defaultdict(list)
    for file_path, date_name, case_id in file_entries:
        cases[(date_name, case_id)].append(file_path)

    print(f"Input directory: {input_dir}/")
    print(f"Output directory: {output_dir}/")
    print(f"Templates directory: {templates_dir}/")
    print()
    print(f"Found {len(file_entries)} file(s) in {len(cases)} case(s):")
    for (date_name, case_id), files in list(cases.items())[:10]:
        print(f"   - {date_name}/{case_id}/: {len(files)} file(s)")
    if len(cases) > 10:
        print(f"   ... and {len(cases) - 10} more cases")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize processor with verbose output
    print("Initializing processor...")
    print()
    try:
        processor = DocumentProcessor(
            templates_dir=templates_dir,
            verbose=True
        )
    except Exception as e:
        print(f"Failed to load templates: {e}")
        print()
        print("Tip: Make sure you have run: python update_configs.py")
        sys.exit(1)

    # Initialize VLM recognizer if enabled
    vlm_recognizer = None
    vlm_results = []

    if args.enable_vlm:
        print()
        print("Initializing Ollama glm-ocr model...")
        print()
        try:
            from vlm_pdf_recognizer.recognition import VLMLoader, VLMRecognizer, TEMPLATE_SCHEMAS

            # Load VLM model with hardware-adaptive loading
            loader = VLMLoader.get_instance()
            client, _, device, model_name = loader.load_model()

            print(f"   Ollama VLM ready:")
            print(f"   Model: {model_name}")
            print(f"   Device: {device}")
            print()

            # Initialize recognizer
            vlm_recognizer = VLMRecognizer(client, None, TEMPLATE_SCHEMAS)

        except ImportError as e:
            print(f"   Warning: VLM dependencies not installed: {e}")
            print(f"   Please install: pip install requests opencv-python numpy")
            print(f"   Continuing without VLM recognition...")
            args.enable_vlm = False
            print()
        except Exception as e:
            print(f"   Warning: Failed to initialize VLM model: {e}")
            print(f"   Continuing without VLM recognition...")
            args.enable_vlm = False
            print()

    print()
    print("=" * 70)
    print("Processing documents...")
    print("=" * 70)
    print()

    # Process all files
    all_results = []

    for idx, (input_file, date_name, case_id) in enumerate(file_entries, 1):
        # Create mirrored output directory: output/date/case_id/
        case_output_dir = os.path.join(output_dir, date_name, case_id)
        os.makedirs(case_output_dir, exist_ok=True)

        print(f"[{idx}/{len(file_entries)}] Processing: {date_name}/{case_id}/{input_file.name}")
        print("-" * 70)

        try:
            # Process file (may contain multiple pages for PDFs)
            results = processor.process_file(str(input_file))

            # Save each result
            for result in results:
                save_result(result, case_output_dir, save_rois=False)
                all_results.append(result)

                # Print summary
                if result.success:
                    print(f"Success: Matched '{result.matched_template_id}' "
                          f"(confidence: {result.confidence_score}, "
                          f"time: {result.processing_time_ms:.0f}ms)")

                    # Run VLM recognition if enabled and ROIs extracted
                    if args.enable_vlm and vlm_recognizer and result.extracted_rois:
                        print(f"   Running VLM recognition on {len(result.extracted_rois)} ROI fields...")
                        try:
                            # Extract ROI images from ExtractedROI objects
                            roi_images = [roi.roi_image for roi in result.extracted_rois]

                            # Process with VLM and preprocessing
                            vlm_output = vlm_recognizer.process_document(
                                roi_images=roi_images,
                                template_id=result.matched_template_id,
                                page_number=result.page_number,
                                document_name=Path(result.input_path).name,
                                blank_template_roi_cache=processor.blank_template_roi_cache,
                                case_id=case_id
                            )

                            vlm_results.append(vlm_output)

                            print(f"   VLM Recognition complete: results={vlm_output.results}, "
                                  f"time={vlm_output.total_processing_time_ms:.0f}ms")

                            # Save original and AIP-processed ROI images
                            try:
                                save_preprocessed_rois(result, vlm_output, case_output_dir)
                                print(f"   Original ROI images saved to: {case_output_dir}/rois/")
                                print(f"   AIP-processed ROI images saved to: {case_output_dir}/processed_rois/")
                            except Exception as roi_error:
                                print(f"   Warning: ROI save failed: {roi_error}")

                            # Save visualization with VLM results (color-coded ROI boxes)
                            try:
                                vis_path = save_vlm_visualization(result, vlm_output, case_output_dir)
                                print(f"   VLM visualization updated: {Path(vis_path).name}")
                            except Exception as vis_error:
                                print(f"   Warning: VLM visualization failed: {vis_error}")

                        except Exception as vlm_error:
                            print(f"   Warning: VLM recognition failed: {vlm_error}")
                            # Create error result for failed VLM recognition
                            error_vlm_output = DocumentRecognitionOutput(
                                document_name=Path(result.input_path).name,
                                page_number=result.page_number,
                                template_id=result.matched_template_id,
                                field_results=[],
                                results=False,
                                processing_timestamp=datetime.now(),
                                total_processing_time_ms=0.0,
                                case_id=case_id
                            )
                            vlm_results.append(error_vlm_output)

                else:
                    print(f"Failed: {result.error_message}")

                    # Create error VLM result for failed preprocessing
                    if args.enable_vlm and vlm_recognizer:
                        error_vlm_output = DocumentRecognitionOutput(
                            document_name=Path(result.input_path).name,
                            page_number=result.page_number,
                            template_id=result.matched_template_id,
                            field_results=[],
                            results=False,
                            processing_timestamp=datetime.now(),
                            total_processing_time_ms=0.0,
                            case_id=case_id
                        )
                        vlm_results.append(error_vlm_output)

        except Exception as e:
            print(f"   Error: {str(e)}")
            # Create error result
            error_result = ProcessingResult(
                input_path=str(input_file),
                page_number=0,
                matched_template_id="error",
                confidence_score=0,
                processing_time_ms=0,
                aligned_image=np.zeros((100, 100, 3), dtype=np.uint8),
                visualization_image=np.zeros((100, 100, 3), dtype=np.uint8),
                extracted_rois=[],
                success=False,
                error_message=str(e)
            )
            all_results.append(error_result)

            # Create error VLM result for exception cases
            if args.enable_vlm and vlm_recognizer:
                error_vlm_output = DocumentRecognitionOutput(
                    document_name=input_file.name,
                    page_number=0,
                    template_id="error",
                    field_results=[],
                    results=False,
                    processing_timestamp=datetime.now(),
                    total_processing_time_ms=0.0,
                    case_id=case_id
                )
                vlm_results.append(error_vlm_output)

        print()

    # Save VLM results per date directory
    print("=" * 70)
    print("Saving VLM results...")
    print("=" * 70)

    # Group results by date
    date_all_results = defaultdict(list)
    date_vlm_results = defaultdict(list)

    for result in all_results:
        # Extract date from output path: input_path contains the date directory
        rel_path = Path(result.input_path)
        # input/date/case_id/file -> date is 2nd component
        parts = rel_path.parts
        # Find the date part (parent of case_id, which is parent of file)
        date_name = parts[-3] if len(parts) >= 3 else "unknown"
        date_all_results[date_name].append(result)

    if args.enable_vlm and vlm_results:
        for vlm_result in vlm_results:
            # Find the date for this vlm_result by matching document_name in file_entries
            date_name = "unknown"
            for file_path, d_name, c_id in file_entries:
                if file_path.name == vlm_result.document_name and c_id == vlm_result.case_id:
                    date_name = d_name
                    break
            date_vlm_results[date_name].append(vlm_result)

    # Save per date directory
    all_dates = set(date_all_results.keys()) | set(date_vlm_results.keys())
    for date_name in sorted(all_dates):
        date_output_dir = os.path.join(output_dir, date_name)
        os.makedirs(date_output_dir, exist_ok=True)

        d_all = date_all_results.get(date_name, [])
        d_vlm = date_vlm_results.get(date_name, [])

        if args.enable_vlm and d_vlm:
            summary_path = save_batch_summary_with_vlm(d_all, d_vlm, date_output_dir)
            print(f"   [{date_name}] VLM_results saved: {summary_path} ({len(d_vlm)} records)")
        else:
            save_batch_summary(d_all, date_output_dir)
            print(f"   [{date_name}] VLM_results saved (preprocessing only)")

    # Print final summary
    print()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)

    success_count = sum(1 for r in all_results if r.success)
    failed_count = len(all_results) - success_count

    print(f"   Summary:")
    print(f"   Total documents: {len(all_results)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {failed_count}")

    if success_count > 0:
        avg_time = sum(r.processing_time_ms for r in all_results if r.success) / success_count
        print(f"   Average time: {avg_time:.0f}ms per document")

    print()
    print(f"   Output directory: {output_dir}/")
    print(f"   - Structure mirrors input: output/date/case_id/")
    if args.enable_vlm and vlm_results:
        print(f"   - Visualizations: *_visualization.png (with color-coded results)")
        print(f"     Green boxes: Content detected (field_id: True)")
        print(f"     Red boxes: No content (field_id: False)")
    else:
        print(f"   - Visualizations: *_visualization.png")
    print(f"   - ROI images: rois/*_roi_*.png")
    print(f"   - Metadata: metadata/*_metadata.json")
    print(f"   - VLM results: {{date}}/VLM_results.json")
    if args.enable_vlm and vlm_results:
        print(f"   - VLM recognition CSV: {{date}}/vlm_recognition_results.csv")
    print()

    # VLM recognition summary with case-level results
    if args.enable_vlm and vlm_results:
        valid_count = sum(1 for r in vlm_results if r.results)
        invalid_count = len(vlm_results) - valid_count
        avg_vlm_time = sum(r.total_processing_time_ms for r in vlm_results) / len(vlm_results)

        print("VLM Recognition Summary:")
        print(f"   Total documents recognized: {len(vlm_results)}")
        print(f"   Valid (results=True): {valid_count}")
        print(f"   Invalid (results=False): {invalid_count}")
        print(f"   Average VLM time: {avg_vlm_time:.0f}ms per document")
        print()

        # Case-level summary (requires all results=True AND all 3 template types present)
        from vlm_pdf_recognizer.output import REQUIRED_TEMPLATE_TYPES
        case_groups = defaultdict(list)
        for r in vlm_results:
            if r.case_id:
                case_groups[r.case_id].append(r)

        if case_groups:
            print("Case-level Results:")
            for case_id, case_docs in sorted(case_groups.items()):
                all_valid = all(r.results for r in case_docs)
                present_types = {r.template_id for r in case_docs}
                missing_types = REQUIRED_TEMPLATE_TYPES - present_types
                case_valid = all_valid and len(missing_types) == 0
                status = "True" if case_valid else "False"
                valid_count = sum(1 for r in case_docs if r.results)
                info = f"{valid_count}/{len(case_docs)} documents valid, types: {sorted(present_types)}"
                if missing_types:
                    info += f", missing: {sorted(missing_types)}"
                print(f"   - {case_id}: {status} ({info})")
            print()

    # Template distribution
    if success_count > 0:
        print("Template distribution:")
        template_counts = {}
        for result in all_results:
            if result.success:
                tid = result.matched_template_id
                template_counts[tid] = template_counts.get(tid, 0) + 1

        for template_id, count in sorted(template_counts.items()):
            print(f"   - {template_id}: {count} document(s)")

    print()

    if failed_count > 0:
        print("Failed documents:")
        for result in all_results:
            if not result.success:
                print(f"   - {Path(result.input_path).name}: {result.error_message}")
        print()

    print("Done!")
    print()

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print()
        print("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print()
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
