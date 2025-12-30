#!/usr/bin/env python3
"""Main program: Process all files in input/ directory."""

import os
import sys
import argparse
from pathlib import Path
from vlm_pdf_recognizer.pipeline import DocumentProcessor
from vlm_pdf_recognizer.output import save_result, save_batch_summary, save_batch_summary_with_vlm, save_vlm_visualization

def main():
    """Process all files in input/ directory and save to output/."""

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
    print("VLM PDF Recognizer - Batch Processing")
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

    # Find all image and PDF files in input directory
    input_path = Path(input_dir)
    input_files = []

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.pdf', '*.JPG', '*.JPEG', '*.PNG', '*.PDF']:
        input_files.extend(input_path.glob(ext))

    input_files = sorted(input_files)

    if not input_files:
        print(f"   No files found in '{input_dir}/' directory!")
        print(f"   Supported formats: .jpg, .jpeg, .png, .pdf")
        sys.exit(1)

    print(f"Input directory: {input_dir}/")
    print(f"Output directory: {output_dir}/")
    print(f"Templates directory: {templates_dir}/")
    print()
    print(f"Found {len(input_files)} file(s) to process:")
    for f in input_files[:10]:  # Show first 10
        print(f"   - {f.name}")
    if len(input_files) > 10:
        print(f"   ... and {len(input_files) - 10} more")
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
        print("Initializing VLM model...")
        print()
        try:
            from vlm_pdf_recognizer.recognition import VLMLoader, VLMRecognizer, TEMPLATE_SCHEMAS

            # Load VLM model with hardware-adaptive loading
            loader = VLMLoader.get_instance()
            model, tokenizer, device, precision = loader.load_model()

            print(f"   VLM Model loaded successfully:")
            print(f"   Device: {device}")
            print(f"   Precision: {precision}")

            # Show auxiliary comparison status
            if processor.blank_roi_cache.get_loaded_count() > 0:
                print(f"   Auxiliary ROI Comparison: ENABLED")
                print(f"   Loaded blank features: {processor.blank_roi_cache.get_loaded_count()} templates")
            else:
                print(f"   Auxiliary ROI Comparison: DISABLED (no blank features found)")
            print()

            # Initialize recognizer
            vlm_recognizer = VLMRecognizer(model, tokenizer, TEMPLATE_SCHEMAS)

        except ImportError as e:
            print(f"   Warning: VLM dependencies not installed: {e}")
            print(f"   Please install: pip install torch transformers Pillow timm")
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

    for idx, input_file in enumerate(input_files, 1):
        print(f"[{idx}/{len(input_files)}] Processing: {input_file.name}")
        print("-" * 70)

        try:
            # Process file (may contain multiple pages for PDFs)
            results = processor.process_file(str(input_file))

            # Save each result
            for result in results:
                save_result(result, output_dir, save_rois=True)
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

                            # Process with VLM (with auxiliary comparison)
                            vlm_output = vlm_recognizer.process_document(
                                roi_images=roi_images,
                                template_id=result.matched_template_id,
                                page_number=result.page_number,
                                document_name=Path(result.input_path).name,
                                blank_roi_cache=processor.blank_roi_cache
                            )

                            vlm_results.append(vlm_output)

                            print(f"   VLM Recognition complete: results={vlm_output.results}, "
                                  f"time={vlm_output.total_processing_time_ms:.0f}ms")

                            # Save visualization with VLM results (color-coded ROI boxes)
                            try:
                                vis_path = save_vlm_visualization(result, vlm_output, output_dir)
                                print(f"   VLM visualization updated: {Path(vis_path).name}")
                            except Exception as vis_error:
                                print(f"   Warning: VLM visualization failed: {vis_error}")

                        except Exception as vlm_error:
                            print(f"   Warning: VLM recognition failed: {vlm_error}")
                            # Create error result for failed VLM recognition
                            from vlm_pdf_recognizer.recognition.vlm_recognizer import DocumentRecognitionOutput
                            from datetime import datetime
                            error_vlm_output = DocumentRecognitionOutput(
                                document_name=Path(result.input_path).name,
                                page_number=result.page_number,
                                template_id=result.matched_template_id,
                                field_results=[],
                                results=False,
                                processing_timestamp=datetime.now(),
                                total_processing_time_ms=0.0
                            )
                            vlm_results.append(error_vlm_output)

                else:
                    print(f"Failed: {result.error_message}")

                    # Create error VLM result for failed preprocessing
                    if args.enable_vlm and vlm_recognizer:
                        from vlm_pdf_recognizer.recognition.vlm_recognizer import DocumentRecognitionOutput
                        from datetime import datetime
                        error_vlm_output = DocumentRecognitionOutput(
                            document_name=Path(result.input_path).name,
                            page_number=result.page_number,
                            template_id=result.matched_template_id,
                            field_results=[],
                            results=False,
                            processing_timestamp=datetime.now(),
                            total_processing_time_ms=0.0
                        )
                        vlm_results.append(error_vlm_output)

        except Exception as e:
            print(f"   Error: {str(e)}")
            # Create error result
            from vlm_pdf_recognizer.pipeline import ProcessingResult
            import numpy as np
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
                from vlm_pdf_recognizer.recognition.vlm_recognizer import DocumentRecognitionOutput
                from datetime import datetime
                error_vlm_output = DocumentRecognitionOutput(
                    document_name=input_file.name,
                    page_number=0,
                    template_id="error",
                    field_results=[],
                    results=False,
                    processing_timestamp=datetime.now(),
                    total_processing_time_ms=0.0
                )
                vlm_results.append(error_vlm_output)

        print()

    # Save VLM results (with VLM results if enabled)
    print("=" * 70)
    print("Saving VLM results...")
    print("=" * 70)

    if args.enable_vlm and vlm_results:
        # Save integrated summary with VLM recognition results
        summary_path = save_batch_summary_with_vlm(all_results, vlm_results, output_dir)
        print(f"   VLM_results saved with recognition results: {summary_path}")
        print(f"   Total VLM records: {len(vlm_results)}")
    else:
        # Save preprocessing-only summary
        save_batch_summary(all_results, output_dir)
        print(f"   VLM_results saved (preprocessing only)")

    # Print final summary
    print()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)

    success_count = sum(1 for r in all_results if r.success)
    failed_count = len(all_results) - success_count

    print(f"   Summary:")
    print(f"   Total documents: {len(all_results)}")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Failed: {failed_count}")

    if success_count > 0:
        avg_time = sum(r.processing_time_ms for r in all_results if r.success) / success_count
        print(f"   Average time: {avg_time:.0f}ms per document")

    print()
    print(f"   Output directory: {output_dir}/")
    print(f"   - Aligned images: *_aligned.png")
    if args.enable_vlm and vlm_results:
        print(f"   - Visualizations: *_visualization.png (with color-coded results)")
        print(f"     • Green boxes: Content detected (aux/vlm=True)")
        print(f"     • Red boxes: No content (aux/vlm=False)")
        print(f"     • Labels: 'aux' (auxiliary comparison) or 'vlm' (VLM recognition)")
    else:
        print(f"   - Visualizations: *_visualization.png")
    print(f"   - ROI images: rois/*_roi_*.png")
    print(f"   - Metadata: *_metadata.json")
    print(f"   - VLM results: VLM_results.json")
    if args.enable_vlm and vlm_results:
        print(f"   - VLM recognition CSV: vlm_recognition_results.csv")
    print()

    # VLM recognition summary
    if args.enable_vlm and vlm_results:
        valid_count = sum(1 for r in vlm_results if r.results)
        invalid_count = len(vlm_results) - valid_count
        avg_vlm_time = sum(r.total_processing_time_ms for r in vlm_results) / len(vlm_results)

        print("VLM Recognition Summary:")
        print(f"   Total documents recognized: {len(vlm_results)}")
        print(f"   ✅ Valid (results=True): {valid_count}")
        print(f"   ❌ Invalid (results=False): {invalid_count}")
        print(f"   Average VLM time: {avg_vlm_time:.0f}ms per document")
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
