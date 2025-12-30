# Data Model: VLM-Based ROI Content Recognition

**Feature**: 002-vlm-roi-recognition
**Date**: 2025-12-23
**Phase**: 1 - Design & Contracts

## Overview

This document defines the data structures and entities for VLM-based ROI field content recognition, including template-to-field mappings, recognition results, and CSV export schemas.

## Core Entities

### 1. FieldSchema

Defines the structure and recognition requirements for a single document field.

**Attributes**:
- `field_id` (str): Unique identifier (e.g., "company_number1", "VX1", "small")
- `field_type` (str): Recognition type - one of: "title", "checkbox", "stamp", "text", "number"
- `template_id` (str): Parent template identifier (contractor_1, contractor_2, enterprise_1)
- `description` (str): Human-readable field description
- `prompt_template` (str): Traditional Chinese VLM prompt with `<image>\n` prefix and JSON output instruction
- `predefined_value` (Optional[str]): For title fields only - predefined text to output without VLM inference

**Relationships**:
- One TemplateSchema contains multiple FieldSchema objects (1:N)
- Each FieldSchema maps to one ExtractedROI from existing pipeline (1:1)

**Validation Rules**:
- `field_type` must be one of: {"title", "checkbox", "stamp", "text", "number"}
- `prompt_template` for non-title fields must contain `<image>\n` prefix
- `prompt_template` must end with JSON output instruction: `{\"has_content\": true/false, \"content_text\": ...}`
- `predefined_value` must be non-null for title fields, null for all other field types

**Example**:

```python
@dataclass
class FieldSchema:
    field_id: str
    field_type: str  # "title" | "checkbox" | "stamp" | "text" | "number"
    template_id: str
    description: str
    prompt_template: str
    predefined_value: Optional[str] = None

    def validate(self):
        """Validate field schema consistency."""
        assert self.field_type in {"title", "checkbox", "stamp", "text", "number"}

        if self.field_type == "title":
            assert self.predefined_value is not None, f"Title field {self.field_id} must have predefined_value"
        else:
            assert self.predefined_value is None, f"Non-title field {self.field_id} must have null predefined_value"
            assert self.prompt_template.startswith("<image>\n"), f"Prompt for {self.field_id} must start with <image>\\n"
            assert '{"has_content"' in self.prompt_template, f"Prompt for {self.field_id} must include JSON output instruction"

    def get_prompt(self) -> str:
        """Get the VLM prompt for this field."""
        if self.field_type == "title":
            return ""  # No VLM inference for titles
        return self.prompt_template
```

---

### 2. TemplateSchema

Defines the complete set of fields for a document template type.

**Attributes**:
- `template_id` (str): Template identifier (contractor_1, contractor_2, enterprise_1)
- `field_schemas` (List[FieldSchema]): Ordered list of field definitions
- `field_count` (int): Total number of fields (15 for contractor_1, 2 for contractor_2, 17 for enterprise_1)

**Relationships**:
- One TemplateSchema per document template type (singleton for each template_id)

**Validation Rules**:
- `field_schemas` order must match ROI extraction order from existing pipeline
- No duplicate `field_id` values within a template
- `field_count` must equal len(field_schemas)

**Example**:

```python
@dataclass
class TemplateSchema:
    template_id: str
    field_schemas: List[FieldSchema]

    @property
    def field_count(self) -> int:
        return len(self.field_schemas)

    def validate(self):
        """Validate template schema consistency."""
        # Check for duplicate field IDs
        field_ids = [f.field_id for f in self.field_schemas]
        assert len(field_ids) == len(set(field_ids)), f"Duplicate field IDs in template {self.template_id}"

        # Validate all fields
        for field in self.field_schemas:
            field.validate()
            assert field.template_id == self.template_id, f"Field {field.field_id} template_id mismatch"

    def get_field_by_id(self, field_id: str) -> Optional[FieldSchema]:
        """Retrieve field schema by field_id."""
        for field in self.field_schemas:
            if field.field_id == field_id:
                return field
        return None
```

---

### 3. RecognitionResult

Stores the VLM recognition output for a single field.

**Attributes**:
- `field_id` (str): Field identifier (links to FieldSchema)
- `has_content` (Optional[bool]): True if content detected, False if empty, None for title fields (not applicable)
- `content_text` (Optional[str]): Extracted text/number if has_content==True, null otherwise
- `raw_response` (str): Raw VLM output for debugging
- `parse_success` (bool): True if JSON parsing succeeded, False if fallback to defaults
- `inference_time_ms` (float): Time taken for VLM inference in milliseconds

**Lifecycle**:
1. Created after VLM inference with raw_response
2. Populated by JSON parser with has_content/content_text
3. If parsing fails, has_content=False, content_text=None, parse_success=False
4. Aggregated into DocumentRecognitionOutput

**Validation Rules**:
- If has_content is True, content_text may be string or null (unreadable but present)
- If has_content is False, content_text must be null
- If has_content is None (title field), content_text must be non-null predefined value

**Example**:

```python
@dataclass
class RecognitionResult:
    field_id: str
    has_content: Optional[bool]  # None for title fields
    content_text: Optional[str]
    raw_response: str
    parse_success: bool
    inference_time_ms: float

    def validate(self):
        """Validate recognition result consistency."""
        if self.has_content is False:
            assert self.content_text is None, f"Field {self.field_id}: has_content=False must have content_text=None"

        if self.has_content is None:  # Title field
            assert self.content_text is not None, f"Title field {self.field_id} must have non-null content_text"
```

---

### 4. DocumentRecognitionOutput

Aggregates all recognition results for a single document page.

**Attributes**:
- `document_name` (str): Input filename (e.g., "contractor_form.pdf")
- `page_number` (int): Page index (0-based)
- `template_id` (str): Matched template (contractor_1, contractor_2, enterprise_1)
- `field_results` (List[RecognitionResult]): Recognition results for all fields in template
- `timestamp` (datetime): Processing timestamp
- `total_processing_time_ms` (float): Total time for all fields in this document

**Relationships**:
- One DocumentRecognitionOutput per document page
- Contains N RecognitionResult objects where N = TemplateSchema.field_count

**Validation Rules**:
- `len(field_results)` must match expected field count for template_id
- `field_results` order must match TemplateSchema.field_schemas order
- All `field_results[i].field_id` must match `TemplateSchema.field_schemas[i].field_id`

**Example**:

```python
from datetime import datetime

@dataclass
class DocumentRecognitionOutput:
    document_name: str
    page_number: int
    template_id: str
    field_results: List[RecognitionResult]
    timestamp: datetime
    total_processing_time_ms: float

    def validate(self, template_schema: TemplateSchema):
        """Validate document output against template schema."""
        assert len(self.field_results) == template_schema.field_count, (
            f"Field count mismatch: expected {template_schema.field_count}, got {len(self.field_results)}"
        )

        for i, (result, schema) in enumerate(zip(self.field_results, template_schema.field_schemas)):
            assert result.field_id == schema.field_id, (
                f"Field ID mismatch at index {i}: expected {schema.field_id}, got {result.field_id}"
            )
            result.validate()

    def to_csv_row(self) -> dict:
        """Convert to flat dictionary for CSV export."""
        row = {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "template_id": self.template_id,
            "processing_timestamp": self.timestamp.isoformat()
        }

        for result in self.field_results:
            row[f"{result.field_id}_has_content"] = result.has_content
            row[f"{result.field_id}_content_text"] = result.content_text

        return row
```

---

### 5. VLMConfig

Configuration for VLM model loading and hardware detection.

**Attributes**:
- `model_name` (str): HuggingFace model identifier (default: "OpenGVLab/InternVL3_5-1B")
- `device` (str): Selected device ("cuda" or "cpu")
- `precision` (str): Model precision ("BF16", "FP16", "INT8", "INT4", "BF16_CPU")
- `vram_gb` (Optional[float]): Available GPU VRAM in GB (if device=="cuda")
- `ram_gb` (Optional[float]): Available system RAM in GB (if device=="cpu")
- `quantization_fallback` (bool): Whether to attempt INT8→INT4→unquantized fallback on CPU OOM (default: True)
- `cache_dir` (Optional[str]): HuggingFace cache directory override (default: None uses ~/.cache/huggingface/)

**Lifecycle**:
1. Created during VLM loader initialization
2. Populated by hardware detection
3. Logged for debugging/performance analysis

**Example**:

```python
@dataclass
class VLMConfig:
    model_name: str = "OpenGVLab/InternVL3_5-1B"
    device: str = ""  # Populated by detection
    precision: str = ""  # Populated by detection
    vram_gb: Optional[float] = None
    ram_gb: Optional[float] = None
    quantization_fallback: bool = True
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Detect hardware and set device/precision."""
        import torch
        import psutil

        if torch.cuda.is_available():
            self.device = "cuda"
            self.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            self.device = "cpu"
            self.ram_gb = psutil.virtual_memory().available / 1e9

    def to_log_dict(self) -> dict:
        """Convert to dictionary for structured logging."""
        return {
            "model": self.model_name,
            "device": self.device,
            "precision": self.precision,
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb
        }
```

---

## Template-to-Field Mappings

### Contractor_1 Template (15 fields)

```python
CONTRACTOR_1_FIELDS = [
    FieldSchema(
        field_id="contractor_1_title",
        field_type="title",
        template_id="contractor_1",
        description="Document title",
        prompt_template="",
        predefined_value="企業負責人電信信評報告之使用授權書"
    ),
    FieldSchema(
        field_id="company_number1",
        field_type="number",
        template_id="contractor_1",
        description="Company registration number (first instance)",
        prompt_template="<image>\n檢查這個數字欄位區域...",  # Full prompt from research.md
        predefined_value=None
    ),
    # ... (13 more fields: VX1, small, person1, company1, VX2, company2, company_number2, person2, person_number1, year, month, date, big)
]
```

### Contractor_2 Template (2 fields)

```python
CONTRACTOR_2_FIELDS = [
    FieldSchema(
        field_id="contractor_2_title",
        field_type="title",
        template_id="contractor_2",
        description="Document title",
        prompt_template="",
        predefined_value="個人資料特定目的外用告知事項暨同意書"
    ),
    FieldSchema(
        field_id="small",
        field_type="stamp",
        template_id="contractor_2",
        description="Small stamp/seal region",
        prompt_template="<image>\n仔細分析這個印章區域...",
        predefined_value=None
    )
]
```

### Enterprise_1 Template (17 fields)

```python
ENTERPRISE_1_FIELDS = [
    FieldSchema(
        field_id="enterprise_1_title",
        field_type="title",
        template_id="enterprise_1",
        description="Document title",
        prompt_template="",
        predefined_value="企業電信信評報告之使用授權書"
    ),
    # ... (16 more fields: company_number1, VX1, big1, small1, person1, company1, VX2, person2, person3, company_number2, address, year, month, date, big2, small2)
]
```

---

## CSV Export Schema

### Column Structure

For a document with N fields, the CSV has 4 + (N×2) columns:

**Metadata Columns (4)**:
1. `document_name` (string)
2. `page_number` (integer)
3. `template_id` (string)
4. `processing_timestamp` (ISO 8601 datetime string)

**Field Result Columns (N×2)**:
For each field `{field_id}`:
- `{field_id}_has_content` (boolean: True/False/None)
- `{field_id}_content_text` (string or null)

**Example CSV Row (Contractor_1)**:

```csv
document_name,page_number,template_id,processing_timestamp,contractor_1_title_has_content,contractor_1_title_content_text,company_number1_has_content,company_number1_content_text,VX1_has_content,VX1_content_text,small_has_content,small_content_text,person1_has_content,person1_content_text,...
contractor_form.pdf,0,contractor_1,2025-12-23T10:30:45.123456,,企業負責人電信信評報告之使用授權書,true,12345678,true,null,true,null,true,陳小明,...
```

**Encoding**: UTF-8 with BOM (utf-8-sig) for Excel compatibility
**Quoting**: QUOTE_NONNUMERIC - quote all non-numeric fields

---

## State Transitions

### Field Recognition Lifecycle

```
[ROI Extracted]
    → [Preprocessing: BGR→RGB→PIL]
    → [VLM Inference: model.chat()]
    → [JSON Parsing: extract has_content/content_text]
    → [Validation: check consistency]
    → [RecognitionResult created]
```

### Document Processing Lifecycle

```
[PDF Input]
    → [Template Matching & Alignment] (existing pipeline)
    → [ROI Extraction] (existing pipeline)
    → [Load Template Schema for matched template_id]
    → [For each ROI: Field Recognition]
    → [Aggregate → DocumentRecognitionOutput]
    → [Export to CSV]
    → [CSV file written to output/]
```

---

## Integration with Existing Pipeline

### Existing ExtractedROI Dataclass (from vlm_pdf_recognizer.extraction.roi_extractor)

```python
@dataclass
class ExtractedROI:
    roi_id: str
    description: str
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    roi_image: np.ndarray  # OpenCV BGR format
    visualization_color: Tuple[int, int, int] = (0, 255, 0)
```

### Mapping Strategy

1. DocumentProcessor.process_image() outputs List[ExtractedROI]
2. VLM recognizer receives List[ExtractedROI] + matched template_id
3. Load TemplateSchema for template_id
4. For each (ExtractedROI, FieldSchema) pair: run VLM inference
5. Aggregate RecognitionResult list → DocumentRecognitionOutput
6. Export to CSV

**Key Constraint**: ExtractedROI.roi_id must match FieldSchema.field_id for correct mapping.

---

## Summary

All data entities defined:

1. ✅ **FieldSchema**: Template→Field mapping with prompts
2. ✅ **TemplateSchema**: Complete field set for each template type
3. ✅ **RecognitionResult**: VLM output for single field
4. ✅ **DocumentRecognitionOutput**: Aggregated results for document page
5. ✅ **VLMConfig**: Hardware detection and model loading parameters
6. ✅ **CSV Schema**: Flattened export format with {field_id}_has_content/{field_id}_content_text columns

Ready for Phase 1: Quickstart documentation and agent context update.
