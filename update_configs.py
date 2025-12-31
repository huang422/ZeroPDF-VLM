#!/usr/bin/env python3
"""Update config.json files from LabelMe annotation JSON files."""

import os
import json
from pathlib import Path
import cv2
import numpy as np


def extract_and_save_blank_rois(template_id: str, template_image: np.ndarray, rois: list) -> int:
    """
    Extract blank template ROI images and save as ORIGINAL PNG files (no preprocessing).

    CRITICAL: Blank ROIs are saved as-is (original BGR format) for direct template difference.
    No preprocessing is applied to ensure exact pixel-level matching with document ROIs.

    Processing Strategy:
    1. Extract ROI from blank template
    2. Save directly as PNG (no conversion, no filtering, no processing)
    3. Result: Raw BGR image for accurate template difference

    Why NO preprocessing:
    - Direct pixel difference is simplest and most reliable
    - Preserves all intensity information for accurate subtraction
    - Document ROIs also use no preprocessing for consistency

    Args:
        template_id: Template identifier (e.g., 'contractor_1')
        template_image: Blank template image (BGR format, uint8, shape (H, W, 3))
        rois: List of ROI dictionaries with 'id' and 'coordinates'

    Returns:
        Number of blank ROI images successfully saved

    Side Effects:
        Creates directory: data/{template_id}/blank_rois/
        Saves ORIGINAL PNG files: {field_id}.png for each ROI
    """
    blank_rois_dir = f'data/{template_id}/blank_rois'
    os.makedirs(blank_rois_dir, exist_ok=True)

    saved_count = 0

    for roi in rois:
        field_id = roi['id']
        coords = roi['coordinates']
        x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']

        # Validate coordinates are within image bounds
        img_height, img_width = template_image.shape[:2]
        if not (0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height):
            print(f"  ⚠ {field_id}: Invalid coordinates ({x1},{y1})-({x2},{y2}), image size {img_width}x{img_height}")
            continue

        # Extract ROI image (keep original BGR format, no processing)
        roi_image = template_image[y1:y2, x1:x2]

        # Save original blank ROI as PNG (lossless compression)
        # No preprocessing - keep raw image for accurate template difference
        blank_roi_path = f'{blank_rois_dir}/{field_id}.png'
        success = cv2.imwrite(blank_roi_path, roi_image)

        if success:
            saved_count += 1
        else:
            print(f"  ⚠ {field_id}: Failed to save blank ROI to {blank_roi_path}")

    return saved_count


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

    # Load blank template image for blank ROI generation
    template_image_path = f'templates/images/{template_id}.jpg'
    if os.path.exists(template_image_path):
        template_image = cv2.imread(template_image_path)
        if template_image is not None:
            # Extract blank template ROIs (Feature 004 - Preprocessing Pipeline)
            print(f"\nGenerating blank template ROIs...")
            blank_roi_count = extract_and_save_blank_rois(template_id, template_image, rois)
            print(f"  → Generated {blank_roi_count} blank ROI images in data/{template_id}/blank_rois/")
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
