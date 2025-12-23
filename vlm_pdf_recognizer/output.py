"""Output saving utilities for processing results."""

import cv2
import json
from pathlib import Path
from typing import List
from vlm_pdf_recognizer.pipeline import ProcessingResult


def save_result(result: ProcessingResult, output_dir: str, save_rois: bool = False):
    """
    Save processing result to output directory.

    Args:
        result: ProcessingResult object
        output_dir: Directory to save outputs
        save_rois: Whether to save individual ROI images

    Output structure:
        output_dir/
            {input_name}_page{n}_aligned.png
            {input_name}_page{n}_visualization.png
            {input_name}_page{n}_metadata.json
            {input_name}_page{n}_roi_{roi_id}.png  (if save_rois=True)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate base filename
    input_name = Path(result.input_path).stem
    page_suffix = f"_page{result.page_number}" if result.page_number > 0 else ""
    base_name = f"{input_name}{page_suffix}"

    # Save aligned image
    aligned_path = output_path / f"{base_name}_aligned.png"
    # cv2.imwrite(str(aligned_path), result.aligned_image)

    # Save visualization image
    vis_path = output_path / f"{base_name}_visualization.png"
    cv2.imwrite(str(vis_path), result.visualization_image)

    # Save metadata as JSON
    metadata = {
        "input_path": result.input_path,
        "page_number": result.page_number,
        "matched_template_id": result.matched_template_id,
        "confidence_score": result.confidence_score,
        "processing_time_ms": result.processing_time_ms,
        "success": result.success,
        "error_message": result.error_message,
        "roi_count": len(result.extracted_rois),
        "rois": [
            {
                "roi_id": roi.roi_id,
                "description": roi.description,
                "bounding_box": {
                    "x1": roi.bounding_box[0],
                    "y1": roi.bounding_box[1],
                    "x2": roi.bounding_box[2],
                    "y2": roi.bounding_box[3]
                }
            }
            for roi in result.extracted_rois
        ]
    }

    metadata_path = output_path / f"{base_name}_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save individual ROI images if requested
    if save_rois:
        for roi in result.extracted_rois:
            roi_path = output_path / f"{base_name}_roi_{roi.roi_id}.png"
            # cv2.imwrite(str(roi_path), roi.roi_image)


def save_batch_summary(results: List[ProcessingResult], output_dir: str):
    """
    Save batch processing summary.

    Args:
        results: List of ProcessingResult objects
        output_dir: Directory to save summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total_count = len(results)
    success_count = sum(1 for r in results if r.success)
    failure_count = total_count - success_count
    avg_time = sum(r.processing_time_ms for r in results) / total_count if total_count > 0 else 0

    # Template distribution
    template_counts = {}
    for result in results:
        tid = result.matched_template_id
        template_counts[tid] = template_counts.get(tid, 0) + 1

    # Build summary
    summary = {
        "total_documents": total_count,
        "successful": success_count,
        "failed": failure_count,
        "average_processing_time_ms": round(avg_time, 2),
        "template_distribution": template_counts,
        "results": [
            {
                "input_path": r.input_path,
                "page_number": r.page_number,
                "template_id": r.matched_template_id,
                "confidence": r.confidence_score,
                "time_ms": round(r.processing_time_ms, 2),
                "success": r.success,
                "error": r.error_message
            }
            for r in results
        ]
    }

    # Save summary
    summary_path = output_path / "batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
