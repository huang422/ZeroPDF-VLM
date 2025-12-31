"""
Blank Template ROI Cache for Preprocessing Pipeline (Feature 004).

This module provides a runtime cache for blank template ROI images used as
reference baselines in the ROI preprocessing pipeline. Blank ROIs are extracted
from template images during configuration and stored as PNG files.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class BlankTemplateROICache:
    """
    Runtime cache for blank template ROI images.

    Loads and caches blank template ROI images from data/{template_id}/blank_rois/
    directories for fast retrieval during preprocessing.

    Attributes:
        _cache: Dict mapping (template_id, field_id) -> np.ndarray (BGR image)
        _loaded_templates: Set of template IDs that have been loaded
    """

    def __init__(self):
        """Initialize empty blank ROI cache."""
        self._cache: Dict[tuple[str, str], np.ndarray] = {}
        self._loaded_templates: set[str] = set()

    def load_blank_rois(self, template_id: str, data_dir: str = "data") -> int:
        """
        Load all blank ROI images for a specific template into cache.

        Args:
            template_id: Template identifier (e.g., 'contractor_1')
            data_dir: Base data directory containing template subdirectories

        Returns:
            Number of blank ROI images successfully loaded

        Side Effects:
            - Populates internal cache with loaded ROI images
            - Logs warnings for missing directories or unreadable images
        """
        blank_rois_dir = Path(data_dir) / template_id / "blank_rois"

        if not blank_rois_dir.exists():
            logger.warning(
                f"Blank ROIs directory not found for template '{template_id}': {blank_rois_dir}"
            )
            return 0

        if not blank_rois_dir.is_dir():
            logger.warning(
                f"Blank ROIs path is not a directory for template '{template_id}': {blank_rois_dir}"
            )
            return 0

        loaded_count = 0

        # Load all PNG files from blank_rois directory
        for png_file in blank_rois_dir.glob("*.png"):
            field_id = png_file.stem  # Filename without extension

            # Read image in BGR format (OpenCV default)
            roi_image = cv2.imread(str(png_file))

            if roi_image is None:
                logger.warning(
                    f"Failed to load blank ROI image: {png_file}"
                )
                continue

            # Validate image format (must be BGR uint8)
            if roi_image.dtype != np.uint8 or len(roi_image.shape) != 3:
                logger.warning(
                    f"Invalid blank ROI image format for {template_id}/{field_id}: "
                    f"dtype={roi_image.dtype}, shape={roi_image.shape}"
                )
                continue

            # Cache the blank ROI
            cache_key = (template_id, field_id)
            self._cache[cache_key] = roi_image
            loaded_count += 1

        # Mark template as loaded
        self._loaded_templates.add(template_id)

        logger.info(
            f"Loaded {loaded_count} blank ROI images for template '{template_id}'"
        )

        return loaded_count

    def get_blank_roi(self, template_id: str, field_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a specific blank ROI image from cache.

        Args:
            template_id: Template identifier
            field_id: Field identifier (e.g., 'signature', 'stamp1')

        Returns:
            Blank ROI image (BGR uint8 np.ndarray) if found, None otherwise
        """
        cache_key = (template_id, field_id)
        return self._cache.get(cache_key)

    def get_loaded_count(self) -> int:
        """
        Get total number of loaded blank ROIs across all templates.

        Returns:
            Total number of blank ROI images in cache
        """
        return len(self._cache)

    def get_loaded_templates(self) -> set[str]:
        """
        Get set of template IDs that have been loaded.

        Returns:
            Set of template ID strings
        """
        return self._loaded_templates.copy()

    def clear(self):
        """Clear all cached blank ROI images."""
        self._cache.clear()
        self._loaded_templates.clear()
        logger.info("Cleared blank template ROI cache")
