# Research & Technical Decisions: VLM Auxiliary ROI Comparison System

**Feature**: 003-vlm-auxiliary-roi-comparison
**Date**: 2025-12-30
**Status**: Complete

## Overview

This document consolidates research findings and technical decisions for implementing the auxiliary ROI comparison system. All NEEDS CLARIFICATION items from the technical context have been resolved.

## Research Areas

### 1. SIFT Feature Extraction and Matching Approach

**Decision**: Reuse existing `feature_extractor.py` and `template_matcher.py` infrastructure

**Rationale**:
- Project already uses SIFT features for template matching with proven reliability
- Existing code handles grayscale conversion, feature extraction, and error cases
- FLANN-based matcher and RANSAC homography estimation already implemented
- Consistent feature extraction ensures comparable results between blank and document ROIs

**Implementation Details**:
- Use `extract_features(image, is_template=False)` from feature_extractor.py
- Reuse `match_features()` and `compute_homography_and_inliers()` from template_matcher.py
- Similarity calculation: `inlier_count / min(doc_feature_count, blank_feature_count)`
- Threshold: 0.6 (60% match) → similarity ≥ 0.6 = empty, < 0.6 = filled

**Alternatives Considered**:
- ORB features: Faster but less accurate, especially for small ROIs with limited texture
- SURF features: Patent-encumbered, not available in OpenCV by default
- Deep learning embeddings: Overkill for binary filled/unfilled classification, adds model dependency

**Validation**: Existing template matching achieves >95% accuracy with SIFT; ROI comparison has similar requirements

---

### 2. Blank ROI Feature Storage Format

**Decision**: Use NumPy .npz format with structured field_id indexing

**Rationale**:
- Consistent with existing template feature storage (data/{template_id}/features.npz)
- NumPy .npz supports multiple arrays with named keys (one per field_id)
- Efficient binary format, small file size (~100-500KB per template)
- Easy to load/save with numpy.savez() and numpy.load()

**Storage Structure**:
```python
# data/{template_id}/blank_roi_features.npz contains:
{
    "contractor_1_title": {"keypoints": [...], "descriptors": [...]},
    "VX1": {"keypoints": [...], "descriptors": [...]},
    "person1": {"keypoints": [...], "descriptors": [...]},
    # ... one entry per field_id
}
```

**Implementation Notes**:
- Keypoints stored as structured array with (x, y, size, angle, response, octave, class_id)
- Descriptors stored as float32 array shape (N, 128) for SIFT
- field_id as dictionary key for O(1) lookup during comparison

**Alternatives Considered**:
- JSON format: Human-readable but inefficient for float arrays, 10x larger files
- SQLite database: Overcomplicated for simple key-value storage, adds dependency
- Separate files per field: More file I/O overhead, harder to manage consistency

---

### 3. Auxiliary Comparison Integration Point

**Decision**: Integrate auxiliary comparison into `VLMRecognizer._recognize_field()` method

**Rationale**:
- Central point where all field recognition happens (title, checkbox, text, number, stamp)
- Already has field schema information to determine field type
- Can conditionally skip VLM inference based on auxiliary result
- Maintains single responsibility: field recognition with multiple strategies

**Integration Flow**:
```
1. Check field type (title/checkbox → skip auxiliary, proceed with existing logic)
2. For other fields: Run auxiliary comparison against blank ROI features
3. If auxiliary_has_content = False (high similarity): Skip VLM, set vlm_has_content = None
4. If auxiliary_has_content = True (low similarity): Proceed with VLM inference
5. Store both auxiliary_has_content and vlm_has_content in RecognitionResult
```

**Alternatives Considered**:
- Separate auxiliary pre-processing step: Duplicates ROI iteration, adds complexity
- Pipeline stage before VLM recognizer: Breaks encapsulation, harder to maintain fallback logic
- Post-processing comparison: Wastes VLM computation on obviously empty fields

---

### 4. Similarity Threshold Selection

**Decision**: Use 0.6 (60% feature match ratio) as the default threshold

**Rationale**:
- Balanced approach: Robust to minor printing variations while detecting content
- Based on template matching experience: <50 inliers = unknown document, >100 inliers = high confidence
- Typical blank ROI has 80-150 SIFT features depending on printed borders/text
- 60% of 100 features = 60 inliers, reasonable buffer above "unknown" threshold

**Tuning Approach**:
- Start with 0.6, log all similarity scores during initial testing
- Analyze false positives (filled fields marked empty) → lower threshold if >5%
- Analyze false negatives (empty fields marked filled) → raise threshold if >5%
- Consider per-field-type thresholds if stamp/signature/text show different distributions

**Alternatives Considered**:
- Absolute inlier count (e.g., 40): Fails on sparse ROIs with few features
- Lower ratio 0.5: More sensitive but higher false positive rate from alignment/printing variations
- Higher ratio 0.7: Misses light signatures/faint stamps

---

### 5. Error Handling and Fallback Strategy

**Decision**: Graceful degradation to VLM-only mode on auxiliary comparison failures

**Rationale**:
- Auxiliary comparison is an optimization, not a requirement
- Missing blank features, feature extraction errors, or matching failures should not block pipeline
- Preserves backward compatibility: VLM-only mode is the original behavior

**Error Handling Logic**:
```python
try:
    auxiliary_has_content = compare_roi_to_blank(...)
except MissingBlankFeatures:
    log.warning("Blank features not found, falling back to VLM-only")
    auxiliary_has_content = None  # Mark as unavailable
except InsufficientFeatures:
    log.warning(f"ROI has too few features for comparison, using VLM")
    auxiliary_has_content = None
except MatchingError:
    log.error(f"Feature matching failed: {error}, falling back to VLM")
    auxiliary_has_content = None

# Always run VLM if auxiliary unavailable
if auxiliary_has_content is None:
    vlm_has_content = run_vlm_inference(...)
```

**Alternatives Considered**:
- Fail-fast on errors: Breaks user workflows, not acceptable for production
- Default to empty (False): Dangerous, could miss critical filled fields
- Default to filled (True): Wastes VLM compute, defeats purpose of optimization

---

### 6. Output Schema Extension

**Decision**: Add parallel auxiliary columns to existing VLM outputs

**Rationale**:
- Maintains backward compatibility: existing columns unchanged
- Enables downstream analysis: compare auxiliary vs VLM accuracy
- Clear naming convention: `auxiliary_{field_id}_has_content` parallel to `{field_id}_has_content`

**JSON Output Structure**:
```json
{
  "document_ID": "sample.pdf",
  "results": true,
  "type": "contractor_1",
  "title": "企業負責人電信信評報告之使用授權書",
  "processing_timestamp": "2025-12-30T10:30:00",
  "fields": {
    "person1": {
      "has_content": true,                    // VLM result
      "content_text": "張三",
      "auxiliary_has_content": true,          // Auxiliary result (NEW)
      "auxiliary_similarity_score": 0.23      // Debug info (NEW)
    },
    "company1": {
      "has_content": false,
      "content_text": null,
      "auxiliary_has_content": false,
      "auxiliary_similarity_score": 0.87
    }
  }
}
```

**CSV Columns** (flattened):
- `person1_has_content`, `person1_content_text` (existing)
- `person1_auxiliary_has_content`, `person1_auxiliary_similarity` (new)

**Alternatives Considered**:
- Replace VLM columns: Loses transparency, can't compare methods
- Nested structure: More complex parsing, harder for Excel/CSV consumers
- Separate auxiliary output file: Harder to correlate results

---

### 7. Validation Logic Priority

**Decision**: Use auxiliary_has_content for results calculation, preserve VLM data

**Rationale**:
- Auxiliary comparison is more accurate for presence detection (pixel-based ground truth)
- VLM can hallucinate or miss faint content (vision model uncertainty)
- Preserving VLM data allows manual review when auxiliary/VLM disagree

**Validation Logic** (from existing system):
```python
def calculate_results_status(self) -> bool:
    # VX1 priority (existing heuristic, not auxiliary)
    vx1_result = next((r for r in self.field_results if r.field_id == "VX1"), None)
    if vx1_result and vx1_result.has_content:  # Uses heuristic result
        return False

    # Date fields OR (using auxiliary)
    date_fields = [r for r in self.field_results if r.field_id in ("year", "month", "date")]
    date_valid = any(r.auxiliary_has_content for r in date_fields) if date_fields else True

    # Other fields AND (using auxiliary, VX1 excluded)
    other_fields = [
        r for r in self.field_results
        if r.auxiliary_has_content is not None  # Exclude title, checkbox
        and r.field_id not in ("VX1", "year", "month", "date")
    ]
    other_fields_valid = all(r.auxiliary_has_content for r in other_fields)

    return date_valid and other_fields_valid
```

**Special Cases**:
- Title fields: auxiliary_has_content = None (skipped), not used in validation
- Checkboxes (VX1, VX2): Use existing heuristic has_content, not auxiliary
- Auxiliary unavailable: Fall back to VLM has_content for that field

---

### 8. Configuration Script Modification

**Decision**: Extend `update_configs.py` to extract blank ROI features after template feature extraction

**Rationale**:
- Centralized configuration: One script for all template setup
- Uses same template images and ROI coordinates as existing config
- Runs once during setup, no runtime overhead

**Implementation Flow**:
```python
def update_config_from_labelme(template_id):
    # Existing: Extract ROI coordinates, save config.json

    # NEW: Load blank template image
    template_image = cv2.imread(f'templates/images/{template_id}.png')

    # NEW: Extract blank ROI features
    blank_roi_features = {}
    for roi in config['rois']:
        x1, y1, x2, y2 = roi['coordinates']
        roi_image = template_image[y1:y2, x1:x2]

        # Save blank ROI image
        roi_path = f'data/{template_id}/rois/blank_{roi["id"]}.png'
        cv2.imwrite(roi_path, roi_image)

        # Extract SIFT features
        keypoints, descriptors = extract_features(roi_image, is_template=True)
        blank_roi_features[roi['id']] = {
            'keypoints': keypoints,
            'descriptors': descriptors
        }

    # NEW: Save blank ROI features
    np.savez(f'data/{template_id}/blank_roi_features.npz', **blank_roi_features)
```

**Alternatives Considered**:
- Separate blank feature extraction script: Duplicates logic, harder to maintain sync
- Runtime extraction from template image: Adds startup overhead, wastes computation
- Manual feature extraction: Error-prone, doesn't scale

---

### 9. Memory and Performance Optimization

**Decision**: Load blank ROI features once at processor initialization, cache in memory

**Rationale**:
- Blank features are static, don't change during processing
- Small memory footprint: ~5MB per template × 3 templates = 15MB total
- Eliminates repeated file I/O during batch processing

**Caching Strategy**:
```python
class DocumentProcessor:
    def __init__(self, templates_dir="data", verbose=False):
        self.templates = load_all_templates(templates_dir)  # Existing
        self.blank_roi_cache = load_blank_roi_features(templates_dir)  # NEW
        # blank_roi_cache = {
        #     "contractor_1": {field_id: (keypoints, descriptors), ...},
        #     "contractor_2": {...},
        #     "enterprise_1": {...}
        # }
```

**Performance Expectations**:
- Initialization overhead: +100-200ms (one-time cost)
- Per-ROI comparison: 30-50ms (SIFT extraction 20ms + matching 10-30ms)
- Total pipeline increase: <20% for typical 10-13 field documents

**Alternatives Considered**:
- Lazy loading per template: Adds latency to first document of each type
- Re-load from disk per document: 100x slower, wasteful I/O
- Global singleton cache: Harder to test, not thread-safe if needed later

---

### 10. Testing Strategy

**Decision**: Three-tier testing approach (unit → integration → backward compatibility)

**Test Coverage**:

**Unit Tests** (`tests/unit/`):
- `test_blank_roi_cache.py`: Feature loading, missing file handling, corrupted data
- `test_roi_comparator.py`: Similarity calculation, threshold logic, edge cases (no features, dimension mismatch)
- `test_vlm_recognizer.py`: Auxiliary integration, field type routing, fallback logic

**Integration Tests** (`tests/integration/`):
- `test_vlm_pipeline.py`: End-to-end with auxiliary enabled (filled/unfilled documents)
- `test_backward_compat.py`: Auxiliary disabled or missing → VLM-only mode works identically

**Test Data Requirements**:
- Blank template images (existing in templates/images/)
- Filled document samples (existing in tests/fixtures/)
- Partially filled documents (mix of empty/filled fields)
- Edge cases: Very faint signatures, light stamps, no features ROIs

**Alternatives Considered**:
- Manual testing only: Not repeatable, doesn't catch regressions
- End-to-end only: Hard to isolate failures, slow test suite
- Mock-heavy unit tests: Doesn't validate actual SIFT matching behavior

---

## Summary of Technical Decisions

| Area | Decision | Key Rationale |
|------|----------|---------------|
| Feature Extraction | Reuse existing SIFT infrastructure | Proven reliability, consistency with template matching |
| Storage Format | NumPy .npz with field_id keys | Efficient binary format, consistent with existing patterns |
| Integration Point | VLMRecognizer._recognize_field() | Central recognition logic, conditional VLM execution |
| Similarity Threshold | 0.6 (60% ratio) | Balanced sensitivity, tunable with logging |
| Error Handling | Graceful degradation to VLM-only | Preserves functionality, backward compatible |
| Output Schema | Parallel auxiliary columns | Transparency, backward compatibility, analysis-friendly |
| Validation Priority | Auxiliary > VLM for has_content | Higher accuracy for presence detection |
| Configuration | Extend update_configs.py | Centralized setup, one-time extraction |
| Memory Management | Load once, cache in memory | 15MB footprint, eliminates I/O overhead |
| Testing | Unit + integration + compat tests | Comprehensive coverage, regression prevention |

## Implementation Readiness

All technical unknowns resolved. Ready to proceed to Phase 1: Design & Contracts.
