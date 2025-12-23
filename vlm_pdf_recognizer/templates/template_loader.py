"""Template loader for golden templates and configurations."""

import os
import json
from typing import List
import cv2

from . import GoldenTemplate, ROI
from . import template_cache


class InvalidTemplateError(Exception):
    """Raised when template directory, image, or config is invalid."""
    pass


def load_template(template_id: str, data_dir: str = "data") -> GoldenTemplate:
    """
    Load a template with image, config, and cached features.

    Args:
        template_id: One of ["enterprise_1", "contractor_1", "contractor_2"]
        data_dir: Base directory for templates (default: "data")

    Returns:
        Loaded GoldenTemplate object with all fields populated

    Raises:
        InvalidTemplateError: Template directory, image, or config missing
    """
    # Define paths - actual structure:
    # templates/images/{template_id}.jpg        - Template image
    # templates/location/{template_id}.json     - LabelMe annotations (for reference)
    # data/{template_id}/config.json            - ROI configuration (auto-generated)
    # data/{template_id}/template_features.pkl  - Cached SIFT features (auto-generated)

    # Template image is in templates/images/
    template_image_path = os.path.join("templates", "images", f"{template_id}.jpg")

    # Config and cache are in data/ subdirectory
    template_dir = os.path.join(data_dir, template_id)
    config_path = os.path.join(template_dir, "config.json")
    features_cache_path = os.path.join(template_dir, "template_features.pkl")

    # Create data directory if it doesn't exist
    os.makedirs(template_dir, exist_ok=True)

    # Check required files
    if not os.path.exists(template_image_path):
        raise InvalidTemplateError(f"Template image not found: {template_image_path}")

    if not os.path.exists(config_path):
        raise InvalidTemplateError(f"Config file not found: {config_path}")

    # Load template image
    template_image = cv2.imread(template_image_path)
    if template_image is None:
        raise InvalidTemplateError(f"Failed to load template image: {template_image_path}")

    image_shape = template_image.shape

    # Load config JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise InvalidTemplateError(f"Invalid JSON in config file: {e}")

    # Parse ROIs
    rois = []
    for roi_data in config.get('rois', []):
        coords = roi_data['coordinates']
        roi = ROI(
            roi_id=roi_data['id'],
            description=roi_data['description'],
            x1=coords['x1'],
            y1=coords['y1'],
            x2=coords['x2'],
            y2=coords['y2'],
            format=roi_data.get('format', 'top_left_bottom_right')
        )

        # Validate ROI coordinates
        if not roi.is_valid(image_shape):
            raise InvalidTemplateError(
                f"ROI '{roi.roi_id}' coordinates out of bounds: "
                f"({roi.x1}, {roi.y1}, {roi.x2}, {roi.y2}) vs image {image_shape}"
            )

        rois.append(roi)

    # Create template object
    template = GoldenTemplate(
        template_id=template_id,
        template_image_path=template_image_path,
        config_path=config_path,
        features_cache_path=features_cache_path,
        image_shape=image_shape,
        rois=rois
    )

    # Load or compute SIFT features
    if template_cache.is_cache_valid(features_cache_path, template_image_path):
        # Load from cache
        try:
            keypoints, descriptors = template_cache.load_features(features_cache_path)
            template.keypoints = keypoints
            template.descriptors = descriptors
        except Exception as e:
            # Cache corrupted, will recompute
            print(f"Warning: Failed to load cache for {template_id}, will recompute: {e}")
            _compute_and_cache_features(template, template_image)
    else:
        # Compute and cache
        _compute_and_cache_features(template, template_image)

    return template


def _compute_and_cache_features(template: GoldenTemplate, template_image) -> None:
    """Compute SIFT features and cache them."""
    # Import here to avoid circular dependency
    from ..alignment import feature_extractor

    keypoints, descriptors = feature_extractor.extract_sift_features(template_image)

    if descriptors is None:
        raise InvalidTemplateError(
            f"No SIFT features found in template: {template.template_id}"
        )

    template.keypoints = keypoints
    template.descriptors = descriptors

    # Cache features
    template_cache.save_features(
        keypoints,
        descriptors,
        template.template_id,
        template.features_cache_path,
        template.image_shape
    )


def load_all_templates(data_dir: str = "data") -> List[GoldenTemplate]:
    """
    Load all three templates.

    Args:
        data_dir: Base directory for templates (default: "data")

    Returns:
        List of 3 GoldenTemplate objects

    Raises:
        InvalidTemplateError: Any template fails to load
    """
    template_ids = ["enterprise_1", "contractor_1", "contractor_2"]
    templates = []

    for template_id in template_ids:
        template = load_template(template_id, data_dir)
        templates.append(template)

    return templates
