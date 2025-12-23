#!/usr/bin/env python3
"""Update config.json files from LabelMe annotation JSON files."""

import os
import json
from pathlib import Path

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


if __name__ == '__main__':
    templates = ['enterprise_1', 'contractor_1', 'contractor_2']

    print("Updating config.json files from LabelMe annotations...\n")

    for template_id in templates:
        update_config_from_labelme(template_id)
        print()

    print("Done!")
