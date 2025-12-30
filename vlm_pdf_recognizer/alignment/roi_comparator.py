"""
ROI Comparator Module

Compares document ROIs against blank template ROIs using SIFT feature matching.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2


def is_roi_blank(roi_image: np.ndarray, brightness_threshold: int = 240, blank_ratio_threshold: float = 0.99) -> bool:
    """
    Check if an ROI image is blank (almost entirely white/empty).

    Args:
        roi_image: ROI image, BGR format, uint8, shape (H, W, 3)
        brightness_threshold: Pixel brightness threshold (0-255), default 240
        blank_ratio_threshold: Ratio of bright pixels required to consider blank (0.0-1.0), default 0.99

    Returns:
        True if ROI is blank (>99% pixels are bright), False otherwise

    Note:
        Uses 99% threshold to avoid false positives when ROI has small amount of text.
        If even 1% of pixels are dark (text/stamps), ROI is considered filled.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # Count bright pixels (close to white)
    bright_pixels = np.sum(gray >= brightness_threshold)
    total_pixels = gray.size

    # Calculate ratio of bright pixels
    blank_ratio = bright_pixels / total_pixels

    return blank_ratio >= blank_ratio_threshold


@dataclass
class ROIComparisonResult:
    """Result of comparing a document ROI against blank template ROI."""
    field_id: str
    similarity_score: float  # Range [0.0, 1.0]
    inlier_count: int
    doc_feature_count: int
    blank_feature_count: int
    auxiliary_has_content: Optional[bool]  # True=filled, False=empty, None=error
    comparison_time_ms: float
    error_message: Optional[str] = None


def compare_roi_to_blank(
    doc_roi_image: np.ndarray,
    blank_features,  # BlankROIFeatures
    similarity_threshold: float = 0.6,
    field_id: str = None
) -> ROIComparisonResult:
    """
    Compare a document ROI image against blank template ROI features.

    Args:
        doc_roi_image: Document ROI image, BGR format, uint8, shape (H, W, 3)
        blank_features: BlankROIFeatures with keypoints and descriptors
        similarity_threshold: Threshold for filled/unfilled determination (default 0.6)
        field_id: Field identifier for logging (optional)

    Returns:
        ROIComparisonResult with similarity score and auxiliary_has_content determination
    """
    import time
    from vlm_pdf_recognizer.alignment.feature_extractor import extract_features
    from vlm_pdf_recognizer.alignment.template_matcher import match_features, compute_homography_and_inliers

    start_time = time.time()
    field_id = field_id or blank_features.field_id

    try:
        # Step 1: Extract SIFT features from document ROI
        try:
            doc_keypoints, doc_descriptors = extract_features(doc_roi_image, is_template=False)
        except ValueError as e:
            # Insufficient features in document ROI (possibly blank)
            # Use pixel-based blank detection
            doc_is_blank = is_roi_blank(doc_roi_image)
            comparison_time = (time.time() - start_time) * 1000

            return ROIComparisonResult(
                field_id=field_id,
                similarity_score=1.0 if doc_is_blank else 0.0,  # 1.0 = similar to blank, 0.0 = different
                inlier_count=0,
                doc_feature_count=0,
                blank_feature_count=blank_features.feature_count,
                auxiliary_has_content=not doc_is_blank,  # False if doc is blank, True if has content
                comparison_time_ms=comparison_time,
                error_message=None  # Successfully determined using pixel-based detection
            )

        doc_feature_count = len(doc_keypoints)
        blank_feature_count = blank_features.feature_count

        # Check minimum feature requirements
        if doc_feature_count < 4:
            # Document ROI has insufficient features (possibly blank)
            # Use pixel-based blank detection
            doc_is_blank = is_roi_blank(doc_roi_image)
            comparison_time = (time.time() - start_time) * 1000

            return ROIComparisonResult(
                field_id=field_id,
                similarity_score=1.0 if doc_is_blank else 0.0,  # 1.0 = similar to blank, 0.0 = different
                inlier_count=0,
                doc_feature_count=doc_feature_count,
                blank_feature_count=blank_feature_count,
                auxiliary_has_content=not doc_is_blank,  # False if doc is blank, True if has content
                comparison_time_ms=comparison_time,
                error_message=None  # Successfully determined using pixel-based detection
            )

        if blank_feature_count < 4:
            # Blank template has insufficient features (pure blank background)
            # Use pixel-based blank detection instead
            doc_is_blank = is_roi_blank(doc_roi_image)
            comparison_time = (time.time() - start_time) * 1000

            return ROIComparisonResult(
                field_id=field_id,
                similarity_score=1.0 if doc_is_blank else 0.0,  # 1.0 = similar to blank, 0.0 = different
                inlier_count=0,
                doc_feature_count=doc_feature_count,
                blank_feature_count=blank_feature_count,
                auxiliary_has_content=not doc_is_blank,  # False if doc is blank, True if has content
                comparison_time_ms=comparison_time,
                error_message=None  # Successfully determined using pixel-based detection
            )

        # Step 2: Match features using FLANN
        try:
            good_matches = match_features(doc_descriptors, blank_features.descriptors, ratio_threshold=0.7)
        except Exception as e:
            comparison_time = (time.time() - start_time) * 1000
            return ROIComparisonResult(
                field_id=field_id,
                similarity_score=0.0,
                inlier_count=0,
                doc_feature_count=doc_feature_count,
                blank_feature_count=blank_feature_count,
                auxiliary_has_content=None,
                comparison_time_ms=comparison_time,
                error_message=f"MATCHING_FAILED: FLANN matcher raised exception: {str(e)}"
            )

        # Step 3: Compute RANSAC homography and inliers
        H, inlier_mask = compute_homography_and_inliers(
            doc_keypoints,
            blank_features.keypoints,
            good_matches,
            ransac_threshold=5.0
        )

        # Check RANSAC result
        if H is None or inlier_mask is None or len(inlier_mask) == 0:
            # RANSAC failed - features too dissimilar, assume filled
            comparison_time = (time.time() - start_time) * 1000
            return ROIComparisonResult(
                field_id=field_id,
                similarity_score=0.0,
                inlier_count=0,
                doc_feature_count=doc_feature_count,
                blank_feature_count=blank_feature_count,
                auxiliary_has_content=True,  # Assume filled when no homography found
                comparison_time_ms=comparison_time,
                error_message=None  # Not an error, just very different
            )

        # Step 4: Calculate similarity score
        inlier_count = int(inlier_mask.ravel().sum())
        min_features = min(doc_feature_count, blank_feature_count)
        similarity_score = inlier_count / min_features if min_features > 0 else 0.0

        # Ensure similarity score is in valid range
        similarity_score = max(0.0, min(1.0, similarity_score))

        # Step 5: Determine auxiliary_has_content
        # similarity >= threshold → unfilled (high similarity to blank)
        # similarity < threshold → filled (low similarity, has content)
        auxiliary_has_content = similarity_score < similarity_threshold

        comparison_time = (time.time() - start_time) * 1000

        return ROIComparisonResult(
            field_id=field_id,
            similarity_score=similarity_score,
            inlier_count=inlier_count,
            doc_feature_count=doc_feature_count,
            blank_feature_count=blank_feature_count,
            auxiliary_has_content=auxiliary_has_content,
            comparison_time_ms=comparison_time,
            error_message=None
        )

    except Exception as e:
        # Unexpected error
        comparison_time = (time.time() - start_time) * 1000
        return ROIComparisonResult(
            field_id=field_id,
            similarity_score=0.0,
            inlier_count=0,
            doc_feature_count=0,
            blank_feature_count=blank_features.feature_count,
            auxiliary_has_content=None,
            comparison_time_ms=comparison_time,
            error_message=f"FEATURE_EXTRACTION_FAILED: {str(e)}"
        )
