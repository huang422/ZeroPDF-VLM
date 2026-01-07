"""CSV exporter for VLM recognition results with preprocessing integration."""

import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def export_recognition_results_to_csv(vlm_results: List, output_dir: str, filename: str = "vlm_recognition_results.csv"):
    """
    Export VLM recognition results to CSV file with flattened columns.

    Creates a CSV with columns:
    - document_ID: Document identifier (name + page)
    - type: Template ID
    - title: Title field content
    - results: Validation status (True/False)
    - processing_timestamp: ISO format timestamp
    - For each field in any template:
        - {field_id}_VLM_has_content: VLM detection result (True/False/None)
        - {field_id}_content_text: Text content (for text/number fields)
        - {field_id}_AIP_has_content: AIP detection result (True/False/None)

    Args:
        vlm_results: List of DocumentRecognitionOutput objects
        output_dir: Directory to save CSV file
        filename: CSV filename (default: vlm_recognition_results.csv)

    Returns:
        Path to created CSV file

    Raises:
        IOError: If CSV write fails
    """
    from .field_schema import TEMPLATE_SCHEMAS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / filename

    if not vlm_results:
        # Create empty CSV with headers only - Match JSON field order
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['document_ID', 'results', 'type', 'title', 'version', 'processing_timestamp'])
        return str(csv_path)

    # Single pass: collect all unique field IDs, field types, and fields with text content
    # Use OrderedDict to preserve template schema order (matching JSON output)
    from collections import OrderedDict
    field_order = OrderedDict()  # Preserves insertion order (template order)
    field_types = {}  # field_id -> field_type mapping
    fields_with_text = set()  # field_ids that have content_text

    for vlm_result in vlm_results:
        template_schema = TEMPLATE_SCHEMAS.get(vlm_result.template_id)
        for field_result in vlm_result.field_results:
            # Skip title field (has_content=None)
            if field_result.has_content is not None:
                # Preserve template order by adding fields as we encounter them
                if field_result.field_id not in field_order:
                    field_order[field_result.field_id] = True

                # Check if this field has text content
                if field_result.content_text is not None:
                    fields_with_text.add(field_result.field_id)

                # Store field type for later use (only once per field_id)
                if field_result.field_id not in field_types and template_schema:
                    field_schema = template_schema.get_field_by_id(field_result.field_id)
                    if field_schema:
                        field_types[field_result.field_id] = field_schema.field_type

    # Use field order from template schema (matches JSON output order)
    sorted_field_ids = list(field_order.keys())

    # Build column headers
    # Match JSON field order: document_ID, results, type, title, version, processing_timestamp
    headers = ['document_ID', 'results', 'type', 'title', 'version', 'processing_timestamp']

    for field_id in sorted_field_ids:
        field_type = field_types.get(field_id, 'unknown')

        # Skip version field - it's now in the base columns (after title)
        if field_type == 'version':
            continue
        # For checkbox/stamp fields: only has_content columns
        elif field_type in ['checkbox', 'stamp']:
            headers.append(f'{field_id}_VLM_has_content')
            headers.append(f'{field_id}_AIP_has_content')
        # For other fields (text/number/person_number): all three columns
        else:
            # Add columns for this field
            headers.append(f'{field_id}_VLM_has_content')
            if field_id in fields_with_text:
                headers.append(f'{field_id}_content_text')
            headers.append(f'{field_id}_AIP_has_content')

    # Build rows
    rows = []
    for vlm_result in vlm_results:
        # Convert to dict for easier access
        result_dict = vlm_result.to_json_dict()

        # Base columns - Match JSON field order: document_ID, results, type, title, version, processing_timestamp
        row = [
            result_dict['document_ID'],
            result_dict['results'],
            result_dict['type'],
            result_dict['title'],
            result_dict.get('version', ''),  # version is now at top level
            result_dict['processing_timestamp']
        ]

        # Field columns
        for field_id in sorted_field_ids:
            field_data = result_dict['fields'].get(field_id, {})
            field_type = field_types.get(field_id, 'unknown')

            # Skip version field - it's already in base columns
            if field_type == 'version':
                continue
            # For checkbox/stamp fields: only has_content columns
            elif field_type in ['checkbox', 'stamp']:
                vlm_has_content = field_data.get('VLM_has_content')
                row.append(vlm_has_content)
                aip_has_content = field_data.get('AIP_has_content')
                row.append(aip_has_content)
            # For other fields: all three columns
            else:
                # VLM has_content
                vlm_has_content = field_data.get('VLM_has_content')
                row.append(vlm_has_content)

                # content_text (if applicable)
                if f'{field_id}_content_text' in headers:
                    content_text = field_data.get('content_text', '')
                    row.append(content_text if content_text else '')

                # AIP has_content
                aip_has_content = field_data.get('AIP_has_content')
                row.append(aip_has_content)

        rows.append(row)

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(rows)

    return str(csv_path)
