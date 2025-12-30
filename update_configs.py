#!/usr/bin/env python3
"""Update config.json files from LabelMe annotation JSON files."""

import os
import json
from pathlib import Path
import cv2
import numpy as np
from vlm_pdf_recognizer.alignment.feature_extractor import extract_features


def extract_blank_roi_features(template_id: str, template_image: np.ndarray, rois: list) -> dict:
    """
    Extract SIFT features from blank template ROIs.

    Args:
        template_id: Template identifier (e.g., 'contractor_1')
        template_image: Blank template image (BGR format)
        rois: List of ROI dictionaries with 'id' and 'coordinates'

    Returns:
        Dictionary mapping field_id to (keypoints, descriptors) tuples
    """
    blank_features = {}
    rois_dir = f'data/{template_id}/rois'
    os.makedirs(rois_dir, exist_ok=True)

    for roi in rois:
        field_id = roi['id']
        coords = roi['coordinates']
        x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']

        # Extract ROI image
        roi_image = template_image[y1:y2, x1:x2]

        # Save blank ROI image for reference
        blank_roi_path = f'{rois_dir}/blank_{field_id}.png'
        cv2.imwrite(blank_roi_path, roi_image)

        # Extract SIFT features
        try:
            keypoints, descriptors = extract_features(roi_image, is_template=True)
            blank_features[field_id] = (keypoints, descriptors)
            print(f"  - {field_id}: {len(keypoints)} features")
        except ValueError as e:
            print(f"  - {field_id}: WARNING - {e}")
            blank_features[field_id] = ([], None)

    return blank_features


def save_blank_roi_features(template_id: str, blank_features: dict):
    """
    Save blank ROI features to .npz file.

    Args:
        template_id: Template identifier
        blank_features: Dictionary mapping field_id to (keypoints, descriptors)
    """
    output_path = f'data/{template_id}/blank_roi_features.npz'

    # Convert keypoints to structured array format for storage
    npz_data = {}
    for field_id, (keypoints, descriptors) in blank_features.items():
        # Store features even if insufficient (empty arrays as markers for pixel-based detection)
        if descriptors is not None and len(keypoints) > 0:
            # Convert KeyPoint objects to structured array
            kp_array = np.array([
                (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in keypoints
            ], dtype=[
                ('x', 'f4'), ('y', 'f4'), ('size', 'f4'), ('angle', 'f4'),
                ('response', 'f4'), ('octave', 'i4'), ('class_id', 'i4')
            ])

            npz_data[f'{field_id}_keypoints'] = kp_array
            npz_data[f'{field_id}_descriptors'] = descriptors
        else:
            # Insufficient features - store empty arrays as marker for pixel-based detection
            empty_kp = np.array([], dtype=[
                ('x', 'f4'), ('y', 'f4'), ('size', 'f4'), ('angle', 'f4'),
                ('response', 'f4'), ('octave', 'i4'), ('class_id', 'i4')
            ])
            empty_desc = np.array([], dtype='f4').reshape(0, 128)  # Shape (0, 128)

            npz_data[f'{field_id}_keypoints'] = empty_kp
            npz_data[f'{field_id}_descriptors'] = empty_desc

    # Save to compressed NumPy archive
    np.savez(output_path, **npz_data)
    print(f"✓ Saved blank ROI features to {output_path}")


def update_config_from_labelme(template_id: str):
    """
    Update config.json from LabelMe annotation file.

    Args:
        template_id: Template identifier (e.g., 'enterprise_1')
    """
    # Read LabelMe annotation from templates/location/
    labelme_path = f'templates/location/{template_id}.json'
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme_data = json.load(f)

    # Extract image dimensions
    image_width = labelme_data['imageWidth']
    image_height = labelme_data['imageHeight']

    # Convert shapes to ROI format
    rois = []
    for shape in labelme_data['shapes']:
        label = shape['label']
        points = shape['points']

        # Extract bounding box coordinates
        # points[0] = [x1, y1], points[1] = [x2, y2]
        x1 = int(points[0][0])
        y1 = int(points[0][1])
        x2 = int(points[1][0])
        y2 = int(points[1][1])

        # Ensure correct order (x1 < x2, y1 < y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        roi = {
            "id": label,
            "description": f"{label.replace('_', ' ').title()} field",
            "coordinates": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "format": "top_left_bottom_right"
        }
        rois.append(roi)

    # Build config
    config = {
        "template_name": template_id,
        "template_version": "1.0",
        "image_dimensions": {
            "width": image_width,
            "height": image_height,
            "unit": "pixels"
        },
        "rois": rois
    }

    # Write config
    config_dir = f'data/{template_id}'
    os.makedirs(config_dir, exist_ok=True)

    config_path = f'{config_dir}/config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✓ Updated {config_path}")
    print(f"  Image: {image_width}x{image_height}")
    print(f"  ROIs: {len(rois)}")

    # Load blank template image for feature extraction
    template_image_path = f'templates/images/{template_id}.jpg'
    if os.path.exists(template_image_path):
        print(f"\nExtracting blank ROI features...")
        template_image = cv2.imread(template_image_path)
        if template_image is not None:
            blank_features = extract_blank_roi_features(template_id, template_image, rois)
            save_blank_roi_features(template_id, blank_features)
        else:
            print(f"  WARNING: Could not load template image from {template_image_path}")
    else:
        print(f"  WARNING: Template image not found at {template_image_path}, skipping blank ROI extraction")


if __name__ == '__main__':
    templates = ['enterprise_1', 'contractor_1', 'contractor_2']

    print("Updating config.json files from LabelMe annotations...\n")

    for template_id in templates:
        update_config_from_labelme(template_id)
        print()

    print("Done!")
