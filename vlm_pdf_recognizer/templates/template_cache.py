"""SIFT feature caching module for golden templates."""

import pickle
import os
from typing import List, Tuple
from datetime import datetime
import numpy as np
import cv2


def save_features(
    keypoints: List[cv2.KeyPoint],
    descriptors: np.ndarray,
    template_id: str,
    cache_path: str,
    image_shape: Tuple[int, int, int]
) -> None:
    """
    Save SIFT features to pickle file with atomic write.

    Args:
        keypoints: List of cv2.KeyPoint objects
        descriptors: SIFT descriptors (Nx128, float32)
        template_id: Template identifier
        cache_path: Path to save pickle file
        image_shape: Shape of template image (height, width, channels)
    """
    # Serialize keypoints (cv2.KeyPoint is not directly picklable)
    keypoints_data = [
        (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ]

    cache_data = {
        'keypoints': keypoints_data,
        'descriptors': descriptors,
        'image_shape': image_shape,
        'template_id': template_id,
        'created_at': datetime.now().isoformat()
    }

    # Atomic write using temp file + rename
    temp_path = cache_path + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, cache_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def load_features(cache_path: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Load SIFT features from pickle file.

    Args:
        cache_path: Path to pickle file

    Returns:
        Tuple of (keypoints, descriptors)

    Raises:
        FileNotFoundError: Cache file doesn't exist
        pickle.UnpicklingError: Cache file corrupted
    """
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    # Reconstruct cv2.KeyPoint objects
    keypoints = [
        cv2.KeyPoint(
            x=pt[0][0],
            y=pt[0][1],
            size=pt[1],
            angle=pt[2],
            response=pt[3],
            octave=pt[4],
            class_id=pt[5]
        )
        for pt in data['keypoints']
    ]

    descriptors = data['descriptors']

    return keypoints, descriptors


def is_cache_valid(cache_path: str, template_image_path: str) -> bool:
    """
    Check if cache is newer than template image.

    Args:
        cache_path: Path to cache file
        template_image_path: Path to template image

    Returns:
        True if cache exists and is newer than image, False otherwise
    """
    if not os.path.exists(cache_path):
        return False

    if not os.path.exists(template_image_path):
        return False

    cache_mtime = os.path.getmtime(cache_path)
    image_mtime = os.path.getmtime(template_image_path)

    return cache_mtime > image_mtime
