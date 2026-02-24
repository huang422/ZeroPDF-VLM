"""CSV exporter for VLM recognition results with preprocessing integration and case-level aggregation."""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional


def export_recognition_results_to_csv(
    vlm_results: List,
    output_dir: str,
    filename: str = "vlm_recognition_results.csv",
    case_aggregated: Optional[Dict[str, Dict[str, Any]]] = None
):
    """
    Export VLM recognition results to CSV file with flattened columns.

    Creates a CSV with columns:
    - case_id: Case identifier from input directory structure
    - case_results: Case-level validation (False if ANY document in case is False)
    - document_ID: Document identifier (name + page)
    - results: Document-level validation status (True/False)
    - type: Template ID
    - title: Title field content
    - version: Version field value
    - processing_timestamp: ISO format timestamp
    - For each field in any template:
        - {field_id}_has_content: AIP detection result (True/False/None)
        - {field_id}_content_text: Text content (for stamp/text/number/person_number fields, from VLM)

    Args:
        vlm_results: List of DocumentRecognitionOutput objects
        output_dir: Directory to save CSV file
        filename: CSV filename (default: vlm_recognition_results.csv)
        case_aggregated: Optional dict of case_id -> {case_results: bool, ...}

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
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                'case_id', 'case_results', 'document_ID', 'results',
                'type', 'title', 'version', 'processing_timestamp'
            ])
        return str(csv_path)

    # Build case_aggregated if not provided
    if case_aggregated is None:
        from collections import defaultdict
        case_groups = defaultdict(list)
        for vlm_result in vlm_results:
            case_id = getattr(vlm_result, 'case_id', None) or "unknown"
            case_groups[case_id].append(vlm_result)

        case_aggregated = {}
        for case_id, results in case_groups.items():
            case_valid = all(r.results for r in results)
            case_aggregated[case_id] = {"case_results": case_valid}

    # Single pass: collect all unique field IDs, field types, and fields with text content
    from collections import OrderedDict
    field_order = OrderedDict()
    field_types = {}
    fields_with_text = set()

    for vlm_result in vlm_results:
        template_schema = TEMPLATE_SCHEMAS.get(vlm_result.template_id)
        for field_result in vlm_result.field_results:
            # Determine field type from schema
            field_schema = None
            if template_schema:
                field_schema = template_schema.get_field_by_id(field_result.field_id)

            # Skip title fields
            if field_schema and field_schema.field_type == "title":
                continue

            if field_result.field_id not in field_order:
                field_order[field_result.field_id] = True

            if field_result.content_text is not None:
                fields_with_text.add(field_result.field_id)

            if field_result.field_id not in field_types and field_schema:
                field_types[field_result.field_id] = field_schema.field_type

    sorted_field_ids = list(field_order.keys())

    # Build column headers with case_id and case_results
    headers = [
        'case_id', 'case_results', 'document_ID', 'results',
        'type', 'title', 'version', 'processing_timestamp'
    ]

    for field_id in sorted_field_ids:
        field_type = field_types.get(field_id, 'unknown')

        # Skip version field - it's in the base columns (after title)
        if field_type == 'version':
            continue
        # For checkbox fields: only has_content (no text recognition)
        elif field_type == 'checkbox':
            headers.append(f'{field_id}_has_content')
        # For stamp/text/number/person_number fields: has_content + content_text
        else:
            headers.append(f'{field_id}_has_content')
            if field_id in fields_with_text:
                headers.append(f'{field_id}_content_text')

    # Build rows
    rows = []
    for vlm_result in vlm_results:
        # Convert to dict for easier access
        result_dict = vlm_result.to_json_dict()

        # Get case info
        case_id = getattr(vlm_result, 'case_id', None) or "unknown"
        case_info = case_aggregated.get(case_id, {})
        case_results = case_info.get("case_results", False)

        # Base columns
        row = [
            case_id,
            case_results,
            result_dict['document_ID'],
            result_dict['results'],
            result_dict['type'],
            result_dict['title'],
            result_dict.get('version', ''),
            result_dict['processing_timestamp']
        ]

        # Field columns
        for field_id in sorted_field_ids:
            field_data = result_dict['fields'].get(field_id, {})
            field_type = field_types.get(field_id, 'unknown')

            # Skip version field - already in base columns
            if field_type == 'version':
                continue
            # For checkbox fields: only has_content (no text recognition)
            elif field_type == 'checkbox':
                row.append(field_data.get('has_content'))
            # For stamp/text/number/person_number fields: has_content + content_text
            else:
                row.append(field_data.get('has_content'))
                if f'{field_id}_content_text' in headers:
                    content_text = field_data.get('content_text', '')
                    row.append(content_text if content_text else '')

        rows.append(row)

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(rows)

    return str(csv_path)
