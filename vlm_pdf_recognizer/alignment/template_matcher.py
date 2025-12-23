"""Template matching using SIFT features and voting mechanism."""

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


class UnknownDocumentError(Exception):
    """Raised when document doesn't match any template (inlier_count < 50)."""
    pass


@dataclass
class FeatureMatch:
    """Feature matching result for one template."""
    template_id: str
    total_matches: int
    good_matches: List[cv2.DMatch]
    inliers: np.ndarray
    inlier_count: int
    confidence: float
    homography_matrix: np.ndarray  # Added homography matrix

    def __repr__(self):
        return (f"FeatureMatch(template={self.template_id}, "
                f"inliers={self.inlier_count}, confidence={self.confidence:.3f})")


@dataclass
class TemplateMatchResult:
    """Result of template matching with voting."""
    matched_template_id: str
    matched_template: object  # GoldenTemplate object
    match_confidence: float
    inlier_count: int
    homography_matrix: np.ndarray
    all_matches: List[FeatureMatch]


def match_features(
    doc_descriptors: np.ndarray,
    template_descriptors: np.ndarray,
    ratio_threshold: float = 0.7
) -> List[cv2.DMatch]:
    """
    Match SIFT features using FLANN-based matcher with Lowe's ratio test.

    Args:
        doc_descriptors: Document SIFT descriptors (Nx128)
        template_descriptors: Template SIFT descriptors (Mx128)
        ratio_threshold: Lowe's ratio test threshold (default: 0.7)

    Returns:
        List of good matches passing ratio test
    """
    # FLANN parameters for SIFT (based on research.md)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN matching with k=2 for ratio test
    matches = flann.knnMatch(doc_descriptors, template_descriptors, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def compute_homography_and_inliers(
    doc_keypoints: List[cv2.KeyPoint],
    template_keypoints: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    ransac_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute homography matrix and identify inliers using RANSAC.

    Args:
        doc_keypoints: Document keypoints
        template_keypoints: Template keypoints
        good_matches: Matches passing ratio test
        ransac_threshold: RANSAC reprojection threshold in pixels (default: 5.0)

    Returns:
        Tuple of (homography_matrix, inlier_mask)
        - homography_matrix: 3x3 matrix or None if failed
        - inlier_mask: Boolean array (1 = inlier, 0 = outlier)
    """
    if len(good_matches) < 4:
        return None, np.array([])

    # Extract point correspondences
    src_pts = np.float32([doc_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([template_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

    return H, mask


def match_single_template(
    doc_keypoints: List[cv2.KeyPoint],
    doc_descriptors: np.ndarray,
    template_keypoints: List[cv2.KeyPoint],
    template_descriptors: np.ndarray,
    template_id: str
) -> FeatureMatch:
    """
    Match document against a single template.

    Args:
        doc_keypoints: Document SIFT keypoints
        doc_descriptors: Document SIFT descriptors
        template_keypoints: Template SIFT keypoints
        template_descriptors: Template SIFT descriptors
        template_id: Template identifier

    Returns:
        FeatureMatch object with matching results
    """
    # Match features
    good_matches = match_features(doc_descriptors, template_descriptors)

    # Compute homography and inliers
    H, inlier_mask = compute_homography_and_inliers(
        doc_keypoints, template_keypoints, good_matches
    )

    # Count inliers
    if inlier_mask is not None and len(inlier_mask) > 0:
        inlier_count = int(inlier_mask.ravel().sum())
    else:
        inlier_count = 0

    # Compute confidence
    confidence = inlier_count / len(good_matches) if len(good_matches) > 0 else 0.0

    return FeatureMatch(
        template_id=template_id,
        total_matches=len(good_matches),
        good_matches=good_matches,
        inliers=inlier_mask if inlier_mask is not None else np.array([]),
        inlier_count=inlier_count,
        confidence=confidence,
        homography_matrix=H
    )


def match_templates(
    doc_keypoints: List[cv2.KeyPoint],
    doc_descriptors: np.ndarray,
    templates: List  # List[GoldenTemplate]
) -> TemplateMatchResult:
    """
    Match document against all templates and select best match using voting.

    Voting mechanism (from spec.md FR-009): Template with most inliers wins.

    Args:
        doc_keypoints: Document SIFT keypoints
        doc_descriptors: Document SIFT descriptors (Nx128)
        templates: List of GoldenTemplate objects with features loaded

    Returns:
        TemplateMatchResult with winning template

    Raises:
        UnknownDocumentError: All templates have inlier_count < 50
    """
    all_matches = []

    # Match against each template
    for template in templates:
        if not template.has_features():
            raise ValueError(f"Template {template.template_id} has no features loaded")

        match = match_single_template(
            doc_keypoints,
            doc_descriptors,
            template.keypoints,
            template.descriptors,
            template.template_id
        )
        all_matches.append(match)

    # Voting: Select template with maximum inlier count (FR-009)
    winner = max(all_matches, key=lambda m: m.inlier_count)

    # Find the corresponding template object
    matched_template = next(t for t in templates if t.template_id == winner.template_id)

    # Check minimum threshold (FR-016)
    if winner.inlier_count < 50:
        match_summary = ", ".join([
            f"{m.template_id}: {m.inlier_count} inliers"
            for m in sorted(all_matches, key=lambda x: x.inlier_count, reverse=True)
        ])
        raise UnknownDocumentError(
            f"Insufficient feature matches. Best match: {match_summary}. "
            f"Need at least 50 inliers."
        )

    return TemplateMatchResult(
        matched_template_id=winner.template_id,
        matched_template=matched_template,
        match_confidence=winner.confidence,
        inlier_count=winner.inlier_count,
        homography_matrix=winner.homography_matrix,
        all_matches=all_matches
    )
