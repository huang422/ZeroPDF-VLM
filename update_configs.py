#!/usr/bin/env python3
"""
Update config.json and field_schema.py from LabelMe annotation JSON files.

This script ensures ROI ordering consistency across three systems:
1. LabelMe JSON (templates/location/{template_id}.json)
2. Config JSON (data/{template_id}/config.json)
3. field_schema.py (vlm_pdf_recognizer/recognition/field_schema.py)

Usage:
    python update_configs.py
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict
from datetime import datetime


# Field type mapping rules based on field_id patterns
FIELD_TYPE_RULES = {
    'title': lambda field_id: 'title' in field_id.lower(),
    'version': lambda field_id: field_id.lower() == 'version',  # Special handling for version field
    'checkbox': lambda field_id: field_id.upper().startswith('VX'),
    'stamp': lambda field_id: field_id.lower() in ['big1', 'small1', 'big', 'small'],
    'person_number': lambda field_id: field_id.lower() == 'person_number1',  # Special handling for ID number
    'number': lambda field_id: any(x in field_id.lower() for x in ['number', 'year', 'month', 'date']),
    'text': lambda field_id: True  # Default fallback
}

# Prompt template key mapping
PROMPT_KEY_MAP = {
    'title': '',
    'version': 'version',  # Special prompt for version field (8 digits)
    'checkbox': 'checkbox',
    'stamp': 'stamp',
    'number': 'number',
    'person_number': 'person_number',  # Special prompt for person_number1 (1 letter + 9 digits)
    'text': 'text'
}

# Predefined values for title fields
TITLE_VALUES = {
    'contractor_1_title': '企業負責人電信信評報告之使用授權書',
    'contractor_2_title': '個人資料特定目的外用告知事項暨同意書',
    'enterprise_1_title': '企業電信信評報告之使用授權書'
}


def infer_field_type(field_id: str) -> str:
    """
    Infer field type from field_id using pattern matching rules.

    Args:
        field_id: Field identifier (e.g., "VX1", "person1", "contractor_1_title")

    Returns:
        Field type: "title" | "checkbox" | "stamp" | "text" | "number"
    """
    for field_type, rule_func in FIELD_TYPE_RULES.items():
        if rule_func(field_id):
            return field_type
    return 'text'  # Default fallback


def generate_field_description(field_id: str, field_type: str) -> str:
    """
    Generate human-readable description for a field.

    Args:
        field_id: Field identifier
        field_type: Field type

    Returns:
        Human-readable description in Traditional Chinese
    """
    if field_type == 'title':
        return '文件標題'
    elif field_type == 'version':
        return '版本號碼欄位'
    elif field_type == 'person_number':
        return '負責人身分證字號'
    elif field_type == 'checkbox':
        num = field_id[-1] if field_id[-1].isdigit() else ""
        return f'同意/不同意勾選框{num}'
    elif field_type == 'stamp':
        if 'big' in field_id.lower():
            return '大印章區域' if '1' in field_id else '大印章/簽名章區域'
        else:
            return '小印章區域' if '1' in field_id else '小印章/簽名章區域'
    elif field_type == 'number':
        if 'year' in field_id.lower():
            return '年份欄位'
        elif 'month' in field_id.lower():
            return '月份欄位'
        elif 'date' in field_id.lower():
            return '日期欄位'
        elif 'company' in field_id.lower():
            num = field_id[-1] if field_id[-1].isdigit() else ''
            return f'統一編號欄位{num}'
        elif 'person' in field_id.lower():
            return '負責人身分證字號'
        else:
            return f'{field_id.replace("_", " ").title()} field'
    elif field_type == 'text':
        if 'person' in field_id.lower():
            num = field_id[-1] if field_id[-1].isdigit() else ''
            return f'負責人姓名欄位{num}'
        elif 'company' in field_id.lower():
            num = field_id[-1] if field_id[-1].isdigit() else ''
            return f'公司名稱欄位{num}'
        elif 'address' in field_id.lower():
            return '公司地址欄位'
        else:
            return f'{field_id.replace("_", " ").title()} field'
    else:
        return f'{field_id.replace("_", " ").title()} field'


def generate_field_schema_entry(field_id: str, template_id: str) -> Dict:
    """
    Generate field schema dictionary for a single field.

    Args:
        field_id: Field identifier
        template_id: Template identifier

    Returns:
        Dictionary containing field schema data
    """
    field_type = infer_field_type(field_id)
    prompt_key = PROMPT_KEY_MAP[field_type]
    predefined_value = TITLE_VALUES.get(field_id) if field_type == 'title' else None
    description = generate_field_description(field_id, field_type)

    return {
        'field_id': field_id,
        'field_type': field_type,
        'template_id': template_id,
        'description': description,
        'prompt_template_key': prompt_key,
        'predefined_value': predefined_value
    }


def generate_field_schema_python_code(templates_data: Dict[str, List[Dict]]) -> str:
    """
    Generate Python code for field_schema.py based on template ROI data.

    Args:
        templates_data: Dict mapping template_id to list of field schema dicts

    Returns:
        Complete Python source code for field_schema.py
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    # Header
    lines.extend([
        '"""',
        'Field schema definitions for template-specific VLM recognition.',
        '',
        'This module defines the structure of document fields, template schemas,',
        'and Traditional Chinese prompt templates for VLM inference.',
        '',
        '*** AUTO-GENERATED FILE - DO NOT EDIT MANUALLY ***',
        f'*** Generated by update_configs.py at {timestamp} ***',
        '*** Run `python update_configs.py` to regenerate this file ***',
        '"""',
        '',
        'from dataclasses import dataclass',
        'from typing import List, Optional, Dict',
        'import logging',
        '',
        'logger = logging.getLogger(__name__)',
        '',
        ''
    ])

    # FieldSchema dataclass
    lines.extend([
        '@dataclass',
        'class FieldSchema:',
        '    """Defines the structure and recognition requirements for a single document field.',
        '',
        '    Attributes:',
        '        field_id: Unique identifier (e.g., "company_number1", "VX1", "small", "version", "person_number1")',
        '        field_type: Recognition type - one of: "title", "version", "checkbox", "stamp", "text", "number", "person_number"',
        '        template_id: Parent template identifier (contractor_1, contractor_2, enterprise_1)',
        '        description: Human-readable field description',
        '        prompt_template_key: Key to lookup prompt in PROMPT_TEMPLATES dict',
        '        predefined_value: For title fields only - predefined text to output without VLM inference',
        '    """',
        '',
        '    field_id: str',
        '    field_type: str  # "title" | "version" | "checkbox" | "stamp" | "text" | "number" | "person_number"',
        '    template_id: str',
        '    description: str',
        '    prompt_template_key: str',
        '    predefined_value: Optional[str] = None',
        '',
        '    def validate(self):',
        '        """Validate field schema consistency.',
        '',
        '        Raises:',
        '            AssertionError: If validation fails',
        '        """',
        '        assert self.field_type in {"title", "version", "checkbox", "stamp", "text", "number", "person_number"}, \\',
        '            f"Invalid field_type: {self.field_type}"',
        '',
        '        if self.field_type == "title":',
        '            assert self.predefined_value is not None, \\',
        '                f"Title field {self.field_id} must have predefined_value"',
        '        else:',
        '            assert self.predefined_value is None, \\',
        '                f"Non-title field {self.field_id} must have null predefined_value"',
        '',
        '    def get_prompt(self, prompt_templates: Dict[str, str]) -> str:',
        '        """Get the VLM prompt for this field.',
        '',
        '        Args:',
        '            prompt_templates: Dictionary mapping prompt_template_key to prompt text',
        '',
        '        Returns:',
        '            VLM prompt string, or empty string for title fields',
        '        """',
        '        if self.field_type == "title":',
        '            return ""  # No VLM inference for titles',
        '',
        '        if self.prompt_template_key not in prompt_templates:',
        '            logger.warning(f"Prompt template key \'{self.prompt_template_key}\' not found, using generic prompt")',
        '            return prompt_templates.get("generic", "")',
        '',
        '        return prompt_templates[self.prompt_template_key]',
        '',
        ''
    ])

    # TemplateSchema dataclass
    lines.extend([
        '@dataclass',
        'class TemplateSchema:',
        '    """Defines the complete set of fields for a document template type.',
        '',
        '    Attributes:',
        '        template_id: Template identifier (contractor_1, contractor_2, enterprise_1)',
        '        field_schemas: Ordered list of field definitions',
        '        title_field_id: Field ID of the title field for this template',
        '    """',
        '',
        '    template_id: str',
        '    field_schemas: List[FieldSchema]',
        '    title_field_id: str',
        '',
        '    @property',
        '    def field_count(self) -> int:',
        '        """Get total number of fields in this template."""',
        '        return len(self.field_schemas)',
        '',
        '    def validate(self):',
        '        """Validate template schema consistency.',
        '',
        '        Raises:',
        '            AssertionError: If validation fails',
        '        """',
        '        # Check for duplicate field IDs',
        '        field_ids = [f.field_id for f in self.field_schemas]',
        '        assert len(field_ids) == len(set(field_ids)), \\',
        '            f"Duplicate field IDs in template {self.template_id}"',
        '',
        '        # Validate title field exists',
        '        title_fields = [f for f in self.field_schemas if f.field_id == self.title_field_id]',
        '        assert len(title_fields) == 1, \\',
        '            f"Template {self.template_id} must have exactly one title field with ID {self.title_field_id}"',
        '        assert title_fields[0].field_type == "title", \\',
        '            f"Title field {self.title_field_id} must have field_type=\'title\'"',
        '',
        '        # Validate all fields',
        '        for field in self.field_schemas:',
        '            field.validate()',
        '            assert field.template_id == self.template_id, \\',
        '                f"Field {field.field_id} template_id mismatch"',
        '',
        '    def get_field_by_id(self, field_id: str) -> Optional[FieldSchema]:',
        '        """Retrieve field schema by field_id.',
        '',
        '        Args:',
        '            field_id: Field identifier to lookup',
        '',
        '        Returns:',
        '            FieldSchema if found, None otherwise',
        '        """',
        '        for field in self.field_schemas:',
        '            if field.field_id == field_id:',
        '                return field',
        '        return None',
        '',
        ''
    ])

    # PROMPT_TEMPLATES (optimized for Ollama glm-ocr)
    # IMPORTANT: glm-ocr is a pure OCR model. Long/complex prompts get echoed back as output.
    # Keep prompts SHORT (1-2 lines max) to avoid prompt echo in results.
    # Post-processing in vlm_recognizer.py handles:
    #   - Prompt echo detection and stripping
    #   - Removing $$ LaTeX markers
    #   - Converting simplified Chinese to traditional Chinese (opencc)
    #   - Formatting numbers (removing spaces)
    lines.extend([
        '# OCR prompt templates optimized for Ollama glm-ocr',
        '# IMPORTANT: glm-ocr is a pure OCR model. Long/complex prompts get echoed back as output.',
        '# Keep prompts SHORT (1-2 lines max) to avoid prompt echo in results.',
        '# Post-processing handles: $$ removal, simplified→traditional Chinese, number formatting.',
        'PROMPT_TEMPLATES: Dict[str, str] = {',
        '    # Version field: 8-digit version number (e.g., 20250417)',
        '    "version": "這是版本號碼，格式為8位數字。請辨識並只輸出8個數字，欄位如果空白或辨識不到內容就輸出空白",',
        '',
        '    # Checkbox field: detect check marks in circle/square',
        '    "checkbox": "請描述框內是否有打勾、打叉或手寫標記",',
        '',
        '    # Stamp/seal field: detect and read stamp content',
        '    "stamp": "請辨識印章上的繁體中文文字",',
        '',
        '    # Text field: handwritten or printed Traditional Chinese text',
        '    "text": "請辨識圖片中的繁體中文文字",',
        '',
        '    # Number field: digits (company number, year, month, date)',
        '    "number": "請辨識圖片中的數字，只輸出數字",',
        '',
        '    # Person number field: Taiwan ID (1 letter + 9 digits)',
        '    "person_number": "請辨識身分證字號，格式為左到右1個英文字母加9個數字",',
        '',
        '    # Generic fallback prompt',
        '    "generic": "請辨識圖片中的繁體中文文字內容",',
        '}',
        '',
        ''
    ])

    # Generate field lists for each template
    for template_id in sorted(templates_data.keys()):
        fields = templates_data[template_id]
        const_name = f"{template_id.upper()}_FIELDS"

        lines.append(f'# {template_id}: {len(fields)} fields')
        lines.append('# IMPORTANT: Order MUST match config.json ROI order for correct zip() alignment')
        lines.append(f'{const_name} = [')

        for field in fields:
            lines.append('    FieldSchema(')
            lines.append(f'        field_id="{field["field_id"]}",')
            lines.append(f'        field_type="{field["field_type"]}",')
            lines.append(f'        template_id="{field["template_id"]}",')
            lines.append(f'        description="{field["description"]}",')
            lines.append(f'        prompt_template_key="{field["prompt_template_key"]}",')
            if field['predefined_value']:
                lines.append(f'        predefined_value="{field["predefined_value"]}"')
            else:
                lines.append('        predefined_value=None')
            lines.append('    ),')

        lines.append(']')
        lines.append('')

    # Generate TEMPLATE_SCHEMAS mapping
    lines.append('# Global template schemas mapping')
    lines.append('TEMPLATE_SCHEMAS: Dict[str, TemplateSchema] = {')
    for template_id in sorted(templates_data.keys()):
        title_field = next((f['field_id'] for f in templates_data[template_id] if f['field_type'] == 'title'), None)
        lines.append(f'    "{template_id}": TemplateSchema(')
        lines.append(f'        template_id="{template_id}",')
        lines.append(f'        field_schemas={template_id.upper()}_FIELDS,')
        lines.append(f'        title_field_id="{title_field}"')
        lines.append('    ),')
    lines.append('}')
    lines.append('')
    lines.append('')

    # Add validation function
    lines.extend([
        '# Validate all template schemas on module load',
        'def _validate_all_schemas():',
        '    """Validate all template schemas on module import."""',
        '    for template_id, schema in TEMPLATE_SCHEMAS.items():',
        '        try:',
        '            schema.validate()',
        '            logger.debug(f"Template schema \'{template_id}\' validated successfully ({schema.field_count} fields)")',
        '        except AssertionError as e:',
        '            logger.error(f"Template schema \'{template_id}\' validation failed: {e}")',
        '            raise',
        '',
        '',
        '# Run validation on import',
        '_validate_all_schemas()',
        ''
    ])

    return '\n'.join(lines)


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


def update_config_from_labelme(template_id: str) -> List[Dict]:
    """
    Update config.json from LabelMe annotation file.

    Args:
        template_id: Template identifier (e.g., 'enterprise_1')

    Returns:
        List of ROI dictionaries (for field_schema generation)
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

    return rois


if __name__ == '__main__':
    templates = ['enterprise_1', 'contractor_1', 'contractor_2']

    print("=" * 80)
    print("Updating config.json and field_schema.py from LabelMe annotations")
    print("=" * 80)
    print()

    # Step 1: Update all config.json files
    print("STEP 1: Updating config.json files...\n")
    all_templates_data = {}

    for template_id in templates:
        rois = update_config_from_labelme(template_id)

        # Generate field schemas for this template
        field_schemas = [generate_field_schema_entry(roi['id'], template_id) for roi in rois]
        all_templates_data[template_id] = field_schemas

        print()

    # Step 2: Generate field_schema.py
    print("STEP 2: Generating field_schema.py...")

    field_schema_path = 'vlm_pdf_recognizer/recognition/field_schema.py'

    # Generate new field_schema.py
    python_code = generate_field_schema_python_code(all_templates_data)

    with open(field_schema_path, 'w', encoding='utf-8') as f:
        f.write(python_code)

    print(f"  ✓ Generated {field_schema_path}")
    print(f"  → Total templates: {len(all_templates_data)}")
    for template_id, fields in all_templates_data.items():
        print(f"    - {template_id}: {len(fields)} fields")

    print()
    print("=" * 80)
    print("✅ All files updated successfully!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  • config.json files: Updated for all templates")
    print("  • blank_rois/*.png: Generated for all templates")
    print("  • field_schema.py: Auto-generated with correct ROI ordering")
    print()
    print("Next steps:")
    print("  1. Verify field_schema.py imports correctly:")
    print("     python -c 'from vlm_pdf_recognizer.recognition.field_schema import TEMPLATE_SCHEMAS'")
    print("  2. Run your recognition pipeline to test the changes")
    print()
