"""Geometric correction using homography and perspective warp."""

import cv2
import numpy as np
from typing import Tuple


class AlignmentError(Exception):
    """Raised when geometric alignment fails."""
    pass


def warp_perspective(
    image: np.ndarray,
    homography: np.ndarray,
    output_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Apply perspective transformation to align document.

    Args:
        image: Input document image (HxWxC, uint8)
        homography: 3x3 transformation matrix
        output_shape: Target dimensions (height, width)

    Returns:
        Aligned image (output_shape[0] x output_shape[1] x C, uint8)

    Raises:
        AlignmentError: If warp fails
    """
    if homography is None or homography.shape != (3, 3):
        raise AlignmentError("Invalid homography matrix")

    try:
        # Apply perspective warp with bilinear interpolation
        aligned = cv2.warpPerspective(
            image,
            homography,
            (output_shape[1], output_shape[0]),  # (width, height)
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # Black borders
        )
    except Exception as e:
        raise AlignmentError(f"Perspective warp failed: {str(e)}")

    return aligned


def align_document_to_template(
    document_image: np.ndarray,
    homography_matrix: np.ndarray,
    template_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Align document image to template using precomputed homography.

    Args:
        document_image: Input document image (HxWxC, uint8)
        homography_matrix: Precomputed 3x3 homography matrix
        template_shape: Target template dimensions (height, width, channels)

    Returns:
        Aligned image matching template dimensions

    Raises:
        AlignmentError: If alignment fails
    """
    # Warp to template dimensions
    aligned = warp_perspective(document_image, homography_matrix, template_shape[:2])

    return aligned
