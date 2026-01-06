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
        # Create empty CSV with headers only
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['document_ID', 'type', 'title', 'results', 'processing_timestamp'])
        return str(csv_path)

    # Collect all unique field IDs across all templates and identify field types
    all_field_ids = set()
    field_types = {}  # field_id -> field_type mapping

    for vlm_result in vlm_results:
        template_schema = TEMPLATE_SCHEMAS.get(vlm_result.template_id)
        for field_result in vlm_result.field_results:
            # Skip title field (has_content=None)
            if field_result.has_content is not None:
                all_field_ids.add(field_result.field_id)
                # Store field type for later use
                if template_schema:
                    field_schema = template_schema.get_field_by_id(field_result.field_id)
                    if field_schema:
                        field_types[field_result.field_id] = field_schema.field_type

    # Sort field IDs for consistent column order
    sorted_field_ids = sorted(all_field_ids)

    # Build column headers
    # version is now at top level (after title), not in fields
    headers = ['document_ID', 'type', 'title', 'version', 'results', 'processing_timestamp']

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
            # Determine if this field is text/number type (has content_text)
            # We need to check across all templates to see if this field has text
            has_text_column = False
            for vlm_result in vlm_results:
                field_result = next((r for r in vlm_result.field_results if r.field_id == field_id), None)
                if field_result and field_result.content_text is not None:
                    has_text_column = True
                    break

            # Add columns for this field
            headers.append(f'{field_id}_VLM_has_content')
            if has_text_column:
                headers.append(f'{field_id}_content_text')
            headers.append(f'{field_id}_AIP_has_content')

    # Build rows
    rows = []
    for vlm_result in vlm_results:
        # Convert to dict for easier access
        result_dict = vlm_result.to_json_dict()

        # Base columns (including version after title)
        row = [
            result_dict['document_ID'],
            result_dict['type'],
            result_dict['title'],
            result_dict.get('version', ''),  # version is now at top level
            result_dict['results'],
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
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(headers)
        writer.writerows(rows)

    return str(csv_path)
