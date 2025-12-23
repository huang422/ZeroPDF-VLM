"""Geometric correction using homography and perspective warp."""

import cv2
import numpy as np
from typing import Tuple


class AlignmentError(Exception):
    """Raised when geometric alignment fails."""
    pass


def compute_homography(
    doc_keypoints,
    template_keypoints,
    good_matches,
    ransac_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute homography matrix using RANSAC.

    Args:
        doc_keypoints: Document keypoints (List[cv2.KeyPoint])
        template_keypoints: Template keypoints (List[cv2.KeyPoint])
        good_matches: Good feature matches (List[cv2.DMatch])
        ransac_threshold: RANSAC pixel tolerance for inliers (default: 5.0)

    Returns:
        Tuple of (homography_matrix, inlier_mask)
        - homography_matrix: 3x3 transformation matrix (float64)
        - inlier_mask: Nx1 boolean array (1 = inlier, 0 = outlier)

    Raises:
        AlignmentError: Insufficient points or homography computation failed
    """
    if len(good_matches) < 4:
        raise AlignmentError(
            f"Insufficient matches for homography: {len(good_matches)} (need >= 4)"
        )

    # Extract point correspondences
    src_pts = np.float32([
        doc_keypoints[m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        template_keypoints[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    # Compute homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

    if H is None:
        raise AlignmentError("Homography computation failed")

    # Verify matrix is invertible
    if np.linalg.det(H) == 0:
        raise AlignmentError("Homography matrix is singular (det = 0)")

    return H, mask


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
