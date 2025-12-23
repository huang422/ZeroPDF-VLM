#!/usr/bin/env python3
"""Main program: Process all files in input/ directory."""

import os
import sys
from pathlib import Path
from vlm_pdf_recognizer.pipeline import DocumentProcessor
from vlm_pdf_recognizer.output import save_result, save_batch_summary

def main():
    """Process all files in input/ directory and save to output/."""

    print("=" * 70)
    print("VLM PDF Recognizer - Batch Processing")
    print("=" * 70)
    print()

    # Configuration
    input_dir = "input"
    output_dir = "output"
    templates_dir = "data"

    # Check input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory '{input_dir}/' not found!")
        print(f"   Please create it and add your documents.")
        sys.exit(1)

    # Find all image and PDF files in input directory
    input_path = Path(input_dir)
    input_files = []

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.pdf', '*.JPG', '*.JPEG', '*.PNG', '*.PDF']:
        input_files.extend(input_path.glob(ext))

    input_files = sorted(input_files)

    if not input_files:
        print(f"❌ No files found in '{input_dir}/' directory!")
        print(f"   Supported formats: .jpg, .jpeg, .png, .pdf")
        sys.exit(1)

    print(f"📁 Input directory: {input_dir}/")
    print(f"📁 Output directory: {output_dir}/")
    print(f"📁 Templates directory: {templates_dir}/")
    print()
    print(f"📄 Found {len(input_files)} file(s) to process:")
    for f in input_files[:10]:  # Show first 10
        print(f"   - {f.name}")
    if len(input_files) > 10:
        print(f"   ... and {len(input_files) - 10} more")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize processor with verbose output
    print("🔧 Initializing processor...")
    print()
    try:
        processor = DocumentProcessor(
            templates_dir=templates_dir,
            verbose=True
        )
    except Exception as e:
        print(f"❌ Failed to load templates: {e}")
        print()
        print("💡 Tip: Make sure you have run: python update_configs.py")
        sys.exit(1)

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
                    print(f"   ✅ Success: Matched '{result.matched_template_id}' "
                          f"(confidence: {result.confidence_score}, "
                          f"time: {result.processing_time_ms:.0f}ms)")
                else:
                    print(f"   ❌ Failed: {result.error_message}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
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

        print()

    # Save batch summary
    print("=" * 70)
    print("Saving batch summary...")
    print("=" * 70)
    save_batch_summary(all_results, output_dir)

    # Print final summary
    print()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)

    success_count = sum(1 for r in all_results if r.success)
    failed_count = len(all_results) - success_count

    print(f"📊 Summary:")
    print(f"   Total documents: {len(all_results)}")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Failed: {failed_count}")

    if success_count > 0:
        avg_time = sum(r.processing_time_ms for r in all_results if r.success) / success_count
        print(f"   ⏱️  Average time: {avg_time:.0f}ms per document")

    print()
    print(f"📁 Output directory: {output_dir}/")
    print(f"   - Aligned images: *_aligned.png")
    print(f"   - Visualizations: *_visualization.png")
    print(f"   - ROI images: *_roi_*.png")
    print(f"   - Metadata: *_metadata.json")
    print(f"   - Batch summary: batch_summary.json")
    print()

    # Template distribution
    if success_count > 0:
        print("📋 Template distribution:")
        template_counts = {}
        for result in all_results:
            if result.success:
                tid = result.matched_template_id
                template_counts[tid] = template_counts.get(tid, 0) + 1

        for template_id, count in sorted(template_counts.items()):
            print(f"   - {template_id}: {count} document(s)")

    print()

    if failed_count > 0:
        print("⚠️  Failed documents:")
        for result in all_results:
            if not result.success:
                print(f"   - {Path(result.input_path).name}: {result.error_message}")
        print()

    print("✨ Done!")
    print()

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print()
        print("⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print()
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
