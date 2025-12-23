"""Watermark removal using HSV thresholding and binarization."""

import cv2
import numpy as np


def remove_watermarks(image: np.ndarray) -> np.ndarray:
    """
    Remove blue, light gray, and light red watermarks via HSV thresholding.

    Preserves black text, table lines, dark blue pen marks, and red stamps.
    Based on research.md: V < 180 for dark content, S > 100 for red stamps.

    Args:
        image: BGR image (HxWx3, uint8)

    Returns:
        Binary image (HxWx1, uint8, values: 0 or 255)
        - Black pixels (0): background, watermarks removed
        - White pixels (255): preserved content
    """
    if image is None or len(image.shape) != 3:
        raise ValueError("Input must be a BGR color image (HxWx3)")

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Create mask for dark content (black text, stamps, pen marks)
    # Keep pixels with low Value (dark colors)
    # Threshold of 180 based on research.md
    dark_mask = cv2.inRange(v, 0, 180)

    # Alternative: Use saturation to preserve red stamps
    # Red has high saturation even when bright
    red_mask = cv2.inRange(s, 100, 255)

    # Combine masks: keep dark OR high-saturation (red stamps)
    content_mask = cv2.bitwise_or(dark_mask, red_mask)

    # Apply morphological operations to clean noise
    # Close small holes, connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)

    # Remove small noise with opening
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)

    # Create binary output (strictly 0 or 255)
    _, binary_output = cv2.threshold(content_mask, 127, 255, cv2.THRESH_BINARY)

    return binary_output


def binarize(image: np.ndarray) -> np.ndarray:
    """
    Convert image to pure binary (0 or 255 only).

    Args:
        image: Grayscale or BGR image

    Returns:
        Binary image (HxW, uint8, values: 0 or 255)
    """
    if len(image.shape) == 3:
        # Convert to grayscale if color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Otsu's thresholding for automatic threshold selection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def preprocess_document(image: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline: watermark removal + binarization.

    Args:
        image: BGR input image

    Returns:
        Binary preprocessed image (HxW, uint8)
    """
    # Remove watermarks (returns binary mask already)
    watermark_removed = remove_watermarks(image)

    # Ensure it's strictly binary
    preprocessed = binarize(watermark_removed)

    return preprocessed
