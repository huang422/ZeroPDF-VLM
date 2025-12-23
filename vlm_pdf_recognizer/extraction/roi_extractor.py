"""ROI extraction and bounding box visualization."""

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


class ROIExtractionError(Exception):
    """Raised when ROI extraction fails."""
    pass


@dataclass
class ExtractedROI:
    """Extracted ROI region from aligned document."""
    roi_id: str
    description: str
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    roi_image: np.ndarray
    visualization_color: Tuple[int, int, int] = (0, 255, 0)  # Green (BGR)


def extract_rois(aligned_image: np.ndarray, roi_configs: List) -> List[ExtractedROI]:
    """
    Extract ROI regions from aligned image based on template configuration.

    Args:
        aligned_image: Aligned document image (HxWxC, uint8)
        roi_configs: List of ROI objects from template

    Returns:
        List of ExtractedROI objects

    Raises:
        ROIExtractionError: If ROI coordinates are out of bounds
    """
    if aligned_image is None or aligned_image.size == 0:
        raise ROIExtractionError("Aligned image is empty")

    extracted_rois = []
    image_shape = aligned_image.shape

    for roi in roi_configs:
        # Validate coordinates
        if not roi.is_valid(image_shape):
            raise ROIExtractionError(
                f"ROI '{roi.roi_id}' coordinates out of bounds: "
                f"({roi.x1}, {roi.y1}, {roi.x2}, {roi.y2}) vs image {image_shape}"
            )

        # Extract ROI region
        roi_image = aligned_image[roi.y1:roi.y2, roi.x1:roi.x2].copy()

        extracted = ExtractedROI(
            roi_id=roi.roi_id,
            description=roi.description,
            bounding_box=(roi.x1, roi.y1, roi.x2, roi.y2),
            roi_image=roi_image
        )

        extracted_rois.append(extracted)

    return extracted_rois


def draw_roi_boxes(
    image: np.ndarray,
    rois: List[ExtractedROI],
    line_thickness: int = 1,
    font_scale: float = 0.4
) -> np.ndarray:
    """
    Draw ROI bounding boxes on image for visualization.

    Args:
        image: Image to draw on (HxWxC, uint8)
        rois: List of ExtractedROI objects
        line_thickness: Box line thickness (default: 1, thinner)
        font_scale: Label font scale (default: 0.4, smaller)

    Returns:
        New image with bounding boxes drawn (HxWxC, uint8)
    """
    # Create copy to avoid modifying original
    output = image.copy()

    for roi in rois:
        x1, y1, x2, y2 = roi.bounding_box
        color = roi.visualization_color

        # Draw thin rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), color, line_thickness)

        # Draw smaller label above box
        label = f"{roi.roi_id}"
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Position label above box (or inside if too close to top)
        label_y = y1 - 5 if y1 > 20 else y1 + label_size[1] + 5

        # Draw smaller label background
        cv2.rectangle(
            output,
            (x1, label_y - label_size[1] - 2),
            (x1 + label_size[0] + 3, label_y + 2),
            color,
            -1  # Filled
        )

        # Draw smaller label text
        cv2.putText(
            output,
            label,
            (x1 + 1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            1
        )

    return output
