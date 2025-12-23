"""SIFT feature extraction module."""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def extract_sift_features(
    image: np.ndarray,
    nfeatures: int = 0,
    for_template: bool = True
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Extract SIFT keypoints and descriptors from an image.

    Based on research.md recommendations:
    - Templates: No feature limit (extract all)
    - Input documents: Limit to 5000 top features for performance

    Args:
        image: Grayscale or BGR image (HxW or HxWx3, uint8)
        nfeatures: Maximum number of features (0 = unlimited)
        for_template: If True, use template parameters; False for input documents

    Returns:
        Tuple of (keypoints, descriptors)
        - keypoints: List of cv2.KeyPoint objects
        - descriptors: numpy array (Nx128, float32) or None if no features

    Raises:
        ValueError: If image is empty or invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty or invalid")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create SIFT detector with optimized parameters
    if for_template:
        # Template: extract all features, standard parameters
        sift = cv2.SIFT_create(
            nfeatures=0,              # No limit
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
    else:
        # Input document: limit features, lower contrast threshold to catch more
        sift = cv2.SIFT_create(
            nfeatures=5000 if nfeatures == 0 else nfeatures,
            nOctaveLayers=3,
            contrastThreshold=0.03,   # Lower to catch more features
            edgeThreshold=10,
            sigma=1.6
        )

    # Detect and compute
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Return empty list if no features found
    if descriptors is None:
        return [], None

    return keypoints, descriptors


def extract_template_features(template_image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT features optimized for template images.

    Args:
        template_image: Template image (BGR or grayscale)

    Returns:
        Tuple of (keypoints, descriptors)

    Raises:
        ValueError: If no features found
    """
    keypoints, descriptors = extract_sift_features(template_image, for_template=True)

    if descriptors is None or len(keypoints) == 0:
        raise ValueError("No SIFT features found in template image")

    return keypoints, descriptors


def extract_document_features(document_image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT features optimized for input documents.

    Args:
        document_image: Input document image (BGR or grayscale)

    Returns:
        Tuple of (keypoints, descriptors)

    Raises:
        ValueError: If no features found
    """
    keypoints, descriptors = extract_sift_features(document_image, for_template=False)

    if descriptors is None or len(keypoints) == 0:
        raise ValueError("No SIFT features found in document image")

    return keypoints, descriptors


def extract_features(image: np.ndarray, is_template: bool = False) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT features from an image (unified interface).

    Args:
        image: Input image (BGR or grayscale)
        is_template: True if extracting from template, False for input document

    Returns:
        Tuple of (keypoints, descriptors)

    Raises:
        ValueError: If no features found
    """
    if is_template:
        return extract_template_features(image)
    else:
        return extract_document_features(image)
