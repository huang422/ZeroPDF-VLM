"""
Blank ROI Feature Cache Module

Loads and caches blank template ROI features for runtime auxiliary comparison.
"""

from dataclasses import dataclass
from typing import Dict, Set, Optional, List
import numpy as np
import cv2


@dataclass
class BlankROIFeatures:
    """SIFT feature data extracted from a blank template ROI."""
    field_id: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray  # Shape (N, 128), dtype float32
    feature_count: int  # Derived: len(keypoints)

    def __post_init__(self):
        """Validate feature data consistency."""
        assert len(self.keypoints) == self.descriptors.shape[0], \
            "Keypoints and descriptors count mismatch"
        assert self.descriptors.shape[1] == 128, \
            "Invalid SIFT descriptor length"
        assert self.feature_count == len(self.keypoints), \
            "Feature count mismatch"


class BlankROIFeatureCache:
    """In-memory cache of blank ROI features for all templates."""

    def __init__(self):
        """Initialize empty cache."""
        self.templates: Dict[str, Dict[str, BlankROIFeatures]] = {}
        self.loaded_templates: Set[str] = set()
        self.failed_templates: Set[str] = set()

    def load_from_directory(self, templates_dir: str) -> None:
        """
        Load all blank ROI features from template data directory.

        Args:
            templates_dir: Path to templates data directory (e.g., "data")
        """
        import os
        import logging

        logger = logging.getLogger(__name__)

        # Scan for all template directories
        if not os.path.exists(templates_dir):
            logger.warning(f"Templates directory not found: {templates_dir}")
            return

        for template_id in os.listdir(templates_dir):
            template_path = os.path.join(templates_dir, template_id)
            if not os.path.isdir(template_path):
                continue

            # Check for blank_roi_features.npz file
            features_file = os.path.join(template_path, 'blank_roi_features.npz')
            if not os.path.exists(features_file):
                logger.warning(f"Blank features not found for template {template_id}")
                self.failed_templates.add(template_id)
                continue

            try:
                # Load .npz file
                npz_data = np.load(features_file, allow_pickle=True)

                # Parse field features
                template_features = {}
                field_ids = set()

                # Extract field_ids from keys (format: {field_id}_keypoints, {field_id}_descriptors)
                for key in npz_data.files:
                    if key.endswith('_keypoints'):
                        field_id = key[:-len('_keypoints')]
                        field_ids.add(field_id)

                # Reconstruct BlankROIFeatures for each field
                for field_id in field_ids:
                    kp_key = f'{field_id}_keypoints'
                    desc_key = f'{field_id}_descriptors'

                    if kp_key not in npz_data or desc_key not in npz_data:
                        logger.error(f"Invalid blank feature data for {template_id}.{field_id}")
                        continue

                    # Load keypoints from structured array
                    kp_array = npz_data[kp_key]
                    keypoints = []
                    for kp_data in kp_array:
                        kp = cv2.KeyPoint(
                            x=float(kp_data['x']),
                            y=float(kp_data['y']),
                            size=float(kp_data['size']),
                            angle=float(kp_data['angle']),
                            response=float(kp_data['response']),
                            octave=int(kp_data['octave']),
                            class_id=int(kp_data['class_id'])
                        )
                        keypoints.append(kp)

                    # Load descriptors
                    descriptors = npz_data[desc_key]

                    # Validate
                    if descriptors.shape[1] != 128:
                        logger.error(f"Invalid SIFT descriptor length for {template_id}.{field_id}")
                        continue

                    # Create BlankROIFeatures
                    try:
                        blank_features = BlankROIFeatures(
                            field_id=field_id,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            feature_count=len(keypoints)
                        )
                        template_features[field_id] = blank_features
                    except AssertionError as e:
                        logger.error(f"Feature validation failed for {template_id}.{field_id}: {e}")
                        continue

                # Store template features
                if template_features:
                    self.templates[template_id] = template_features
                    self.loaded_templates.add(template_id)
                    logger.info(f"Loaded blank features for {template_id}: {len(template_features)} fields")
                else:
                    logger.warning(f"No valid blank features found for {template_id}")
                    self.failed_templates.add(template_id)

            except Exception as e:
                logger.error(f"Failed to load blank features for {template_id}: {e}")
                self.failed_templates.add(template_id)

    def get_features(
        self,
        template_id: str,
        field_id: str
    ) -> Optional[BlankROIFeatures]:
        """
        Retrieve blank ROI features for a specific template and field.

        Args:
            template_id: Template identifier (e.g., "contractor_1")
            field_id: Field identifier (e.g., "person1")

        Returns:
            BlankROIFeatures if found, None otherwise
        """
        # Check if template exists
        if template_id not in self.templates:
            return None

        # Check if field exists
        if field_id not in self.templates[template_id]:
            return None

        return self.templates[template_id][field_id]

    def has_features(self, template_id: str) -> bool:
        """
        Check if a template has blank features available.

        Args:
            template_id: Template identifier

        Returns:
            True if template was successfully loaded, False otherwise
        """
        return template_id in self.loaded_templates

    def get_loaded_count(self) -> int:
        """Get count of successfully loaded templates."""
        return len(self.loaded_templates)
