# Data Model: VLM Auxiliary ROI Comparison System

**Feature**: 003-vlm-auxiliary-roi-comparison
**Date**: 2025-12-30

## Overview

This document defines the data structures, entities, and relationships for the auxiliary ROI comparison system. All entities integrate with existing VLM pipeline structures.

## Core Entities

### 1. BlankROIFeatures

Represents the SIFT feature data extracted from a blank template ROI.

**Attributes**:
- `field_id` (string): ROI field identifier (e.g., "person1", "company1", "big")
- `keypoints` (array of KeyPoint): SIFT keypoints detected in blank ROI
  - Each KeyPoint contains: (x, y, size, angle, response, octave, class_id)
- `descriptors` (numpy array): SIFT descriptors, shape (N, 128), dtype float32
- `feature_count` (int): Number of features (N), derived from descriptors.shape[0]

**Storage Format**:
- File: `data/{template_id}/blank_roi_features.npz`
- Structure: NumPy compressed archive with one entry per field_id
- Size: ~100-500KB per template (3 templates × 500KB = ~1.5MB total on disk)

**Validation Rules**:
- field_id must match a valid ROI ID from template configuration
- keypoints and descriptors must have same length N
- descriptors must be float32 array with shape (N, 128)
- feature_count must be > 0 (at least one SIFT feature)

**Relationships**:
- Belongs to: One Template (template_id)
- Used by: ROIComparator for similarity calculation

---

### 2. BlankROIFeatureCache

In-memory cache of all blank ROI features for all templates, loaded at processor initialization.

**Attributes**:
- `templates` (dict): Mapping of template_id → field features
  - Structure: `{"contractor_1": {"person1": BlankROIFeatures, ...}, ...}`
- `loaded_templates` (set): Set of template IDs successfully loaded
- `failed_templates` (set): Set of template IDs that failed to load (fallback to VLM-only)

**Methods**:
- `load_from_directory(templates_dir)`: Load all blank features from data/ directory
- `get_features(template_id, field_id)`: Retrieve features for specific field, returns None if missing
- `has_features(template_id)`: Check if template has blank features available

**Initialization**:
```python
cache = BlankROIFeatureCache()
cache.load_from_directory("data")
# Loaded: contractor_1 (13 fields), contractor_2 (2 fields), enterprise_1 (14 fields)
# Total memory: ~15MB
```

**Error Handling**:
- Missing .npz file: Log warning, add template to failed_templates
- Corrupted data: Log error, skip that template
- Missing field_id: Return None from get_features(), caller handles fallback

**Relationships**:
- Contains: Multiple BlankROIFeatures (one per template × field)
- Used by: DocumentProcessor, ROIComparator

---

### 3. ROIComparisonResult

Result of comparing a document ROI against its corresponding blank template ROI.

**Attributes**:
- `field_id` (string): ROI field identifier
- `similarity_score` (float): Inlier count / min(doc_features, blank_features), range [0.0, 1.0]
- `inlier_count` (int): Number of RANSAC inliers from feature matching
- `doc_feature_count` (int): Number of SIFT features in document ROI
- `blank_feature_count` (int): Number of SIFT features in blank ROI
- `auxiliary_has_content` (bool): Content presence determination
  - True if similarity_score < 0.6 (potential content)
  - False if similarity_score >= 0.6 (high similarity to blank, unfilled)
- `comparison_time_ms` (float): Time taken for comparison in milliseconds
- `error_message` (string or None): Error description if comparison failed

**Validation Rules**:
- similarity_score must be in range [0.0, 1.0]
- inlier_count must be <= min(doc_feature_count, blank_feature_count)
- If error_message is not None, auxiliary_has_content should be None (comparison failed)

**Usage**:
```python
result = compare_roi_to_blank(doc_roi_image, blank_features, threshold=0.6)
# result.similarity_score = 0.87
# result.inlier_count = 78
# result.doc_feature_count = 95
# result.blank_feature_count = 89
# result.auxiliary_has_content = False (0.87 >= 0.6)
```

**Relationships**:
- Input: Document ROI image (numpy array)
- Input: BlankROIFeatures for corresponding field
- Output: Used to populate RecognitionResult.auxiliary_has_content

---

### 4. RecognitionResult (Extended)

Existing entity from VLM recognition, extended with auxiliary comparison data.

**New Attributes**:
- `auxiliary_has_content` (bool or None): Auxiliary comparison result
  - True: Low similarity to blank, potential content detected
  - False: High similarity to blank, field appears unfilled
  - None: Auxiliary comparison skipped (title/checkbox) or failed
- `auxiliary_similarity_score` (float or None): Debug metric for tuning threshold
- `auxiliary_comparison_time_ms` (float or None): Performance tracking

**Existing Attributes** (unchanged):
- `field_id` (string)
- `has_content` (bool or None): VLM inference result
- `content_text` (string or None): Extracted text from VLM
- `raw_response` (string): VLM raw output
- `parse_success` (bool): VLM JSON parsing success
- `inference_time_ms` (float): VLM inference time
- `retry_count` (int): VLM retry attempts

**Field Type Behavior**:

| Field Type | auxiliary_has_content | has_content (VLM/heuristic) | content_text |
|------------|----------------------|---------------------------|--------------|
| Title | None (skipped) | None (not applicable) | Predefined title string |
| Checkbox | None (skipped) | Heuristic result (True/False) | Description or None |
| Text/Number | Comparison result (True/False) | VLM result or None (skipped if aux=False) | VLM extraction or None |
| Stamp | Comparison result (True/False) | VLM result or None (skipped if aux=False) | None (stamps don't extract text) |

**Validation Rules** (extended):
- If auxiliary_has_content is False and VLM was skipped, has_content should be None
- If auxiliary_has_content is True, has_content should have VLM result
- Title fields: auxiliary_has_content = None, content_text = predefined value
- Checkbox fields: auxiliary_has_content = None, has_content = heuristic result

---

### 5. DocumentRecognitionOutput (Extended)

Existing entity for document-level results, validation logic updated to use auxiliary results.

**Modified Methods**:

**`calculate_results_status()` - Updated Logic**:
```python
def calculate_results_status(self) -> bool:
    """Calculate results using auxiliary_has_content with fallback to VLM has_content."""

    # 1. VX1 Priority (checkbox uses existing heuristic, not auxiliary)
    vx1_result = next((r for r in self.field_results if r.field_id == "VX1"), None)
    if vx1_result and vx1_result.has_content:  # VX1 heuristic result
        return False  # Disagreement checkbox marked

    # 2. Date Fields OR (use auxiliary_has_content, fallback to VLM has_content)
    date_fields = [r for r in self.field_results if r.field_id in ("year", "month", "date")]
    date_valid = True
    if date_fields:
        date_valid = any(
            r.auxiliary_has_content if r.auxiliary_has_content is not None else r.has_content
            for r in date_fields
        )

    # 3. Other Fields AND (use auxiliary_has_content, fallback to VLM has_content)
    excluded_ids = {"VX1", "year", "month", "date"}
    other_fields = [
        r for r in self.field_results
        if r.has_content is not None  # Exclude title fields
        and r.field_id not in excluded_ids
    ]
    other_fields_valid = all(
        r.auxiliary_has_content if r.auxiliary_has_content is not None else r.has_content
        for r in other_fields
    )

    return date_valid and other_fields_valid
```

**New Attributes** (output schema):
- JSON: Each field in `fields` dictionary gains `auxiliary_has_content` and `auxiliary_similarity_score`
- No changes to existing attributes (document_ID, results, type, title, processing_timestamp, total_processing_time_ms)

**Example Output** (JSON):
```json
{
  "document_ID": "sample.pdf",
  "results": true,
  "type": "contractor_1",
  "title": "企業負責人電信信評報告之使用授權書",
  "processing_timestamp": "2025-12-30T10:30:00",
  "fields": {
    "person1": {
      "has_content": true,
      "content_text": "張三",
      "auxiliary_has_content": true,
      "auxiliary_similarity_score": 0.23
    },
    "company1": {
      "has_content": null,
      "content_text": null,
      "auxiliary_has_content": false,
      "auxiliary_similarity_score": 0.87
    },
    "VX1": {
      "has_content": false,
      "auxiliary_has_content": null
    }
  }
}
```

---

## Data Flow

### Configuration Phase (update_configs.py)

```
1. Load blank template image from templates/images/{template_id}.png
2. Load ROI coordinates from templates/location/{template_id}.json (existing)
3. For each ROI coordinate:
   a. Extract ROI image from blank template
   b. Save ROI image to data/{template_id}/rois/blank_{field_id}.png
   c. Extract SIFT features using feature_extractor.extract_features()
   d. Store keypoints and descriptors in BlankROIFeatures
4. Save all BlankROIFeatures to data/{template_id}/blank_roi_features.npz
```

### Processing Phase (main.py → pipeline.py → vlm_recognizer.py)

```
1. Processor Initialization:
   a. Load templates (existing)
   b. Load BlankROIFeatureCache from data/ directory (NEW)
   c. Initialize VLM model (existing)

2. Per Document:
   a. Template matching and alignment (existing)
   b. ROI extraction (existing)
   c. For each ROI field:
      i. Check field type
      ii. If title: Skip auxiliary, use predefined text
      iii. If checkbox: Skip auxiliary, use heuristic
      iv. If text/number/stamp:
          - Get blank features from cache
          - Compare doc ROI to blank → ROIComparisonResult
          - If auxiliary_has_content = False: Skip VLM, set has_content = None
          - If auxiliary_has_content = True: Run VLM inference
      v. Create RecognitionResult with both auxiliary and VLM data
   d. Calculate results status using auxiliary-priority logic
   e. Generate DocumentRecognitionOutput

3. Output Generation:
   a. Save visualization with auxiliary-based colors (existing, modified)
   b. Save JSON with auxiliary fields (existing, modified)
   c. Save CSV with auxiliary columns (existing, modified)
```

### Fallback Flow (Missing Blank Features)

```
1. BlankROIFeatureCache.load_from_directory() encounters missing .npz file
2. Log warning: "Blank features not found for template_id=contractor_1"
3. Add template_id to failed_templates set
4. During processing:
   a. cache.get_features(template_id, field_id) returns None
   b. VLMRecognizer detects None, sets auxiliary_has_content = None
   c. Proceeds with VLM inference (existing behavior)
   d. Validation logic uses VLM has_content (fallback)
5. User sees warning in logs, system continues functioning (VLM-only mode)
```

---

## State Transitions

### ROI Comparison State Machine

```
[Start] → [Check Field Type]
          ↓
          ├─ Title → [Skip Auxiliary] → [Use Predefined Text] → [Done]
          ├─ Checkbox → [Skip Auxiliary] → [Use Heuristic] → [Done]
          └─ Text/Number/Stamp → [Get Blank Features]
                                  ↓
                                  ├─ Not Found → [Set auxiliary=None] → [Run VLM] → [Done]
                                  └─ Found → [Extract Doc Features]
                                             ↓
                                             ├─ Extraction Failed → [Set auxiliary=None] → [Run VLM]
                                             └─ Success → [Match Features]
                                                          ↓
                                                          ├─ Matching Failed → [Set auxiliary=None] → [Run VLM]
                                                          └─ Success → [Calculate Similarity]
                                                                       ↓
                                                                       ├─ similarity >= 0.6 → [Set auxiliary=False] → [Skip VLM]
                                                                       └─ similarity < 0.6 → [Set auxiliary=True] → [Run VLM]
```

---

## Constraints and Invariants

### Data Integrity

1. **Feature Consistency**: Blank ROI features must use same SIFT parameters as document ROI extraction
   - Implementation: Use feature_extractor.extract_features() for both
   - Validation: feature_count > 0 for all loaded blank features

2. **Field ID Matching**: BlankROIFeatures field_id must match template ROI configuration
   - Implementation: Validate during load_from_directory()
   - Error handling: Log warning, skip invalid fields

3. **Threshold Invariant**: 0.0 <= similarity_score <= 1.0
   - Implementation: Enforce in ROIComparisonResult validation
   - Error handling: Clip to range if calculation exceeds bounds

### Output Completeness

1. **All Fields Present**: DocumentRecognitionOutput must have RecognitionResult for every template field
   - Validation: len(field_results) == template_schema.field_count
   - Error: Raise AssertionError if mismatch

2. **Auxiliary Data Availability**: auxiliary_has_content may be None (skipped/failed) but must be present in output
   - Implementation: Default to None in RecognitionResult constructor
   - CSV: Output "null" for None values

3. **Results Calculation**: Must handle mixed auxiliary/VLM data gracefully
   - Fallback: Use VLM has_content when auxiliary_has_content is None
   - Validation: At least one of (auxiliary_has_content, has_content) must be non-None for non-title fields

### Performance Constraints

1. **Memory Limit**: BlankROIFeatureCache must stay under 50MB
   - Current: ~15MB for 3 templates × ~29 fields total
   - Headroom: 35MB for future template additions

2. **Comparison Time**: Per-ROI auxiliary comparison must complete in <100ms (p95)
   - Typical: 30-50ms (SIFT extraction 20ms + matching 10-30ms)
   - Timeout: If comparison exceeds 500ms, log warning and fallback to VLM

---

## Summary

This data model extends the existing VLM recognition pipeline with auxiliary ROI comparison while maintaining backward compatibility. Key design principles:

- **Minimal Disruption**: Reuse existing SIFT infrastructure, extend existing entities
- **Graceful Degradation**: Missing auxiliary features → automatic VLM-only fallback
- **Transparency**: Both auxiliary and VLM results preserved in outputs
- **Type Safety**: Clear validation rules and invariants for all entities
- **Performance**: In-memory caching, <20% pipeline overhead
