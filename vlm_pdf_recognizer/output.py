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
            rois/
                {input_name}_page{n}_roi_{roi_id}.png  (if save_rois=True)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate base filename
    input_name = Path(result.input_path).stem
    page_suffix = f"_page{result.page_number}" if result.page_number > 0 else ""
    base_name = f"{input_name}{page_suffix}"

    # Save aligned image
    # aligned_path = output_path / f"{base_name}_aligned.png"
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

    # Create metadata subdirectory
    metadata_dir = output_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_dir / f"{base_name}_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save individual ROI images if requested
    if save_rois:
        # Create ROI subdirectory
        roi_dir = output_path / "rois"
        roi_dir.mkdir(parents=True, exist_ok=True)

        for roi in result.extracted_rois:
            roi_path = roi_dir / f"{base_name}_roi_{roi.roi_id}.png"
            cv2.imwrite(str(roi_path), roi.roi_image)


def save_preprocessed_rois(result: ProcessingResult, vlm_output, output_dir: str):
    """
    Save both original and AIP-processed ROI images.

    Args:
        result: ProcessingResult object with extracted_rois
        vlm_output: DocumentRecognitionOutput with field_results containing processed_roi_image
        output_dir: Directory to save ROI images

    Output structure:
        output_dir/
            rois/                           # Original ROI images (extracted, before AIP)
                {input_name}_page{n}_roi_{roi_id}.png
            processed_rois/                 # AIP ROI images (after AIP pipeline)
                {input_name}_page{n}_roi_{roi_id}_processed.png
    """
    output_path = Path(output_dir)

    # Create directories
    original_roi_dir = output_path / "rois"
    processed_roi_dir = output_path / "processed_rois"
    original_roi_dir.mkdir(parents=True, exist_ok=True)
    processed_roi_dir.mkdir(parents=True, exist_ok=True)

    # Generate base filename
    input_name = Path(result.input_path).stem
    page_suffix = f"_page{result.page_number}" if result.page_number > 0 else ""
    base_name = f"{input_name}{page_suffix}"

    # Create a mapping from field_id to field_result
    field_results_dict = {fr.field_id: fr for fr in vlm_output.field_results}

    # Save both original and AIP-processed ROI images
    for roi in result.extracted_rois:
        field_result = field_results_dict.get(roi.roi_id)

        # Save original ROI image (extracted, before any processing)
        original_roi_path = original_roi_dir / f"{base_name}_roi_{roi.roi_id}.png"
        cv2.imwrite(str(original_roi_path), roi.roi_image)

        # Save AIP-processed image if available (final binary from AIP pipeline)
        if field_result and field_result.processed_roi_image is not None:
            processed_roi_path = processed_roi_dir / f"{base_name}_roi_{roi.roi_id}_processed.png"
            cv2.imwrite(str(processed_roi_path), field_result.processed_roi_image)


def save_vlm_visualization(result: ProcessingResult, vlm_output, output_dir: str):
    """
    Save visualization image with VLM recognition results (color-coded ROI boxes).

    Args:
        result: ProcessingResult object with aligned_image and extracted_rois
        vlm_output: DocumentRecognitionOutput with VLM field_results
        output_dir: Directory to save visualization

    Color coding (updated with AIP priority):
        - Green (BGR: 0,255,0): AIP_has_content=True OR has_content=True (filled fields)
        - Red (BGR: 0,0,255): AIP_has_content=False OR has_content=False (empty fields)
        - Blue (BGR: 255,0,0): AIP_has_content=None (AIP error/skip)
        - Original template color: Title fields (has_content=None)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    input_name = Path(result.input_path).stem
    page_suffix = f"_page{result.page_number}" if result.page_number > 0 else ""
    base_name = f"{input_name}{page_suffix}"

    # Create visualization with color-coded ROI boxes
    vis_image = result.aligned_image.copy()

    # Create a mapping from field_id to field_result for correct matching
    field_results_dict = {fr.field_id: fr for fr in vlm_output.field_results}

    # Draw ROI boxes with colors based on VLM results
    for roi in result.extracted_rois:
        x1, y1, x2, y2 = roi.bounding_box

        # Find the matching field_result by field_id
        field_result = field_results_dict.get(roi.roi_id)

        if field_result is None:
            # ROI has no matching field_result (shouldn't happen, but use original color)
            color = roi.visualization_color
            label = f"{roi.roi_id}: N/A"
        else:
            # Color logic (updated with AIP priority - Feature 004):
            # Priority: AIP_has_content (if available) > has_content (VLM inference)
            # - Title fields (has_content=None): Keep original template color
            # - Checkbox fields: Use AIP_has_content if available, fallback to VLM has_content
            # - Other fields: Use AIP_has_content (green=True, red=False, blue=None/error)

            # Check if this is a title field (has_content=None indicates title)
            if field_result.has_content is None:
                # Title field - use original template color
                color = roi.visualization_color
                label = f"{roi.roi_id}: title"
            else:
                # Determine content detection result using AIP
                if field_result.AIP_has_content is not None:
                    # AIP result available
                    has_content = field_result.AIP_has_content
                    if has_content:
                        color = (0, 255, 0)  # GREEN
                        label = f"{roi.roi_id}: True"
                    else:
                        color = (0, 0, 255)  # RED
                        label = f"{roi.roi_id}: False"
                elif field_result.AIP_has_content is None and field_result.AIP_time_ms is not None:
                    # AIP ran but returned None (error case)
                    color = (255, 0, 0)  # BLUE
                    label = f"{roi.roi_id}: ERROR"
                else:
                    # Fallback to VLM result (checkbox heuristic or VLM inference)
                    has_content = field_result.has_content
                    if has_content:
                        color = (0, 255, 0)  # GREEN
                        label = f"{roi.roi_id}: True"
                    else:
                        color = (0, 0, 255)  # RED
                        label = f"{roi.roi_id}: False"

        # Draw rectangle with thickness 2-3 for visibility
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw label with has_content status
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        # Background for label
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - baseline - 2),
                    (x1 + label_size[0], y1), color, -1)

        # Text label
        cv2.putText(vis_image, label, (x1, y1 - baseline - 2),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Save VLM visualization (overwrites the original visualization)
    vis_path = output_path / f"{base_name}_visualization.png"
    cv2.imwrite(str(vis_path), vis_image)

    return str(vis_path)


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
    summary_path = output_path / "VLM_results.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_vlm_recognition_results(vlm_results: List, output_dir: str, filename: str = "vlm_recognition_results.json"):
    """
    Save VLM recognition results to JSON file.

    Args:
        vlm_results: List of DocumentRecognitionOutput objects
        output_dir: Directory to save JSON
        filename: JSON filename (default: vlm_recognition_results.json)

    Raises:
        ValueError: If vlm_results list is empty
        IOError: If file writing fails
    """
    if not vlm_results:
        raise ValueError("Cannot export empty VLM results list to JSON")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert results to list of dictionaries
    results_data = [result.to_json_dict() for result in vlm_results]

    # Write to JSON with UTF-8 encoding
    json_path = output_path / filename
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    return str(json_path)


def save_batch_summary_with_vlm(results: List[ProcessingResult], vlm_results: List, output_dir: str):
    """
    Save batch processing summary with integrated VLM recognition results.

    Args:
        results: List of ProcessingResult objects from preprocessing pipeline
        vlm_results: List of DocumentRecognitionOutput objects from VLM recognition
        output_dir: Directory to save summary

    Note:
        This combines preprocessing pipeline results with VLM recognition results
        into a single VLM_results.json file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate preprocessing statistics
    total_count = len(results)
    success_count = sum(1 for r in results if r.success)
    failure_count = total_count - success_count
    avg_time = sum(r.processing_time_ms for r in results) / total_count if total_count > 0 else 0

    # Template distribution
    template_counts = {}
    for result in results:
        tid = result.matched_template_id
        template_counts[tid] = template_counts.get(tid, 0) + 1

    # Calculate VLM statistics
    vlm_count = len(vlm_results) if vlm_results else 0
    vlm_success_count = sum(1 for v in vlm_results if v.results) if vlm_results else 0
    vlm_failure_count = vlm_count - vlm_success_count
    vlm_avg_time = sum(v.total_processing_time_ms for v in vlm_results) / vlm_count if vlm_count > 0 else 0

    # Convert VLM results to dictionaries
    vlm_data = [v.to_json_dict() for v in vlm_results] if vlm_results else []

    # Build integrated summary
    summary = {
        "preprocessing": {
            "total_documents": total_count,
            "successful": success_count,
            "failed": failure_count,
            "average_processing_time_ms": round(avg_time, 2),
            "template_distribution": template_counts
        },
        "vlm_recognition": {
            "total_documents": vlm_count,
            "successful": vlm_success_count,
            "failed": vlm_failure_count,
            "average_processing_time_ms": round(vlm_avg_time, 2)
        },
        "documents": vlm_data
    }

    # Save integrated summary
    summary_path = output_path / "VLM_results.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Export to CSV with AIP columns
    try:
        from vlm_pdf_recognizer.recognition.csv_exporter import export_recognition_results_to_csv
        csv_path = export_recognition_results_to_csv(vlm_results, output_dir)
    except Exception as csv_error:
        # CSV export is optional - continue if it fails
        print(f"   Warning: CSV export failed: {csv_error}")

    return str(summary_path)
