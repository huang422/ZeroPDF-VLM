# Quickstart: VLM Auxiliary ROI Comparison System

**Feature**: 003-vlm-auxiliary-roi-comparison
**Audience**: Developers implementing this feature
**Date**: 2025-12-30

## Overview

This guide provides a step-by-step walkthrough for implementing and testing the VLM auxiliary ROI comparison system. Follow these steps in order to ensure proper integration with the existing pipeline.

---

## Prerequisites

Before starting implementation:

1. **Read the specification**: [spec.md](spec.md)
2. **Review research decisions**: [research.md](research.md)
3. **Understand data model**: [data-model.md](data-model.md)
4. **Review API contracts**: [contracts/](contracts/)
5. **Ensure existing tests pass**: `pytest tests/` should have 100% pass rate

---

## Implementation Phases

### Phase 1: Blank ROI Feature Extraction (Configuration)

**Goal**: Extend update_configs.py to extract and store blank ROI features

**Files to Modify**:
- `update_configs.py`

**Implementation Steps**:

1. **Add blank ROI extraction function**:
```python
def extract_blank_roi_features(template_id: str, template_image: np.ndarray, rois: List[dict]):
    """Extract SIFT features from blank template ROIs."""
    from vlm_pdf_recognizer.alignment.feature_extractor import extract_features

    blank_roi_features = {}

    # Create ROI images directory
    roi_dir = f'data/{template_id}/rois'
    os.makedirs(roi_dir, exist_ok=True)

    for roi in rois:
        field_id = roi['id']
        coords = roi['coordinates']
        x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']

        # Extract ROI from template image
        roi_image = template_image[y1:y2, x1:x2]

        # Save blank ROI image
        roi_path = f'{roi_dir}/blank_{field_id}.png'
        cv2.imwrite(roi_path, roi_image)

        # Extract SIFT features
        keypoints, descriptors = extract_features(roi_image, is_template=True)

        # Store features
        blank_roi_features[field_id] = {
            'keypoints': keypoints,
            'descriptors': descriptors
        }

    return blank_roi_features
```

2. **Modify update_config_from_labelme()**:
```python
def update_config_from_labelme(template_id: str):
    # ... existing code to save config.json ...

    # NEW: Load template image
    template_image = cv2.imread(f'templates/images/{template_id}.png')

    # NEW: Extract blank ROI features
    blank_features = extract_blank_roi_features(template_id, template_image, rois)

    # NEW: Save blank ROI features
    features_path = f'data/{template_id}/blank_roi_features.npz'
    np.savez(features_path, **blank_features)

    print(f"✓ Saved blank ROI features: {len(blank_features)} fields")
```

3. **Test configuration script**:
```bash
python update_configs.py
# Expected output:
# ✓ Updated data/contractor_1/config.json
# ✓ Saved blank ROI features: 13 fields
# ✓ Updated data/contractor_2/config.json
# ✓ Saved blank ROI features: 2 fields
# ✓ Updated data/enterprise_1/config.json
# ✓ Saved blank ROI features: 14 fields
```

4. **Verify output files**:
```bash
ls -lh data/contractor_1/blank_roi_features.npz
# Should show ~200-500KB file

ls data/contractor_1/rois/
# Should show blank_{field_id}.png for all 13 fields
```

**Unit Tests**:
- Test blank feature extraction with valid template image
- Test handling of missing template image
- Test handling of invalid ROI coordinates

---

### Phase 2: Blank ROI Feature Cache (Runtime Loading)

**Goal**: Implement BlankROIFeatureCache to load and cache blank features

**Files to Create**:
- `vlm_pdf_recognizer/alignment/blank_roi_cache.py`

**Implementation Steps**:

1. **Create data structures**:
```python
@dataclass
class BlankROIFeatures:
    field_id: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    feature_count: int

    def __post_init__(self):
        assert len(self.keypoints) == self.descriptors.shape[0]
        assert self.descriptors.shape[1] == 128
```

2. **Implement BlankROIFeatureCache class**:
```python
class BlankROIFeatureCache:
    def __init__(self):
        self.templates: Dict[str, Dict[str, BlankROIFeatures]] = {}
        self.loaded_templates: Set[str] = set()
        self.failed_templates: Set[str] = set()

    def load_from_directory(self, templates_dir: str) -> None:
        # Implementation as per contract
        pass

    def get_features(self, template_id: str, field_id: str) -> Optional[BlankROIFeatures]:
        # Implementation as per contract
        pass

    def has_features(self, template_id: str) -> bool:
        return template_id in self.loaded_templates
```

3. **Integrate into DocumentProcessor**:
```python
# In vlm_pdf_recognizer/pipeline.py

class DocumentProcessor:
    def __init__(self, templates_dir: str = "data", verbose: bool = False):
        # ... existing code ...

        # NEW: Load blank ROI features
        from vlm_pdf_recognizer.alignment.blank_roi_cache import BlankROIFeatureCache
        self.blank_roi_cache = BlankROIFeatureCache()
        self.blank_roi_cache.load_from_directory(templates_dir)

        if self.verbose:
            print(f"Loaded blank features: {len(self.blank_roi_cache.loaded_templates)} templates")
```

**Unit Tests** (`tests/unit/test_blank_roi_cache.py`):
- test_load_valid_features()
- test_load_missing_file()
- test_get_features_exists()
- test_get_features_missing()

---

### Phase 3: ROI Comparator (Auxiliary Comparison Logic)

**Goal**: Implement compare_roi_to_blank() function

**Files to Create**:
- `vlm_pdf_recognizer/alignment/roi_comparator.py`

**Implementation Steps**:

1. **Create ROIComparisonResult dataclass**:
```python
@dataclass
class ROIComparisonResult:
    field_id: str
    similarity_score: float
    inlier_count: int
    doc_feature_count: int
    blank_feature_count: int
    auxiliary_has_content: Optional[bool]
    comparison_time_ms: float
    error_message: Optional[str] = None
```

2. **Implement compare_roi_to_blank()**:
```python
def compare_roi_to_blank(
    doc_roi_image: np.ndarray,
    blank_features: BlankROIFeatures,
    similarity_threshold: float = 0.6,
    field_id: str = None
) -> ROIComparisonResult:
    start_time = time.time()

    try:
        # 1. Extract SIFT features from document ROI
        from vlm_pdf_recognizer.alignment.feature_extractor import extract_features
        doc_keypoints, doc_descriptors = extract_features(doc_roi_image, is_template=False)

        # 2. Match features
        from vlm_pdf_recognizer.alignment.template_matcher import match_features, compute_homography_and_inliers
        good_matches = match_features(doc_descriptors, blank_features.descriptors)

        # 3. Compute inliers
        H, inlier_mask = compute_homography_and_inliers(
            doc_keypoints, blank_features.keypoints, good_matches
        )

        inlier_count = int(inlier_mask.ravel().sum()) if inlier_mask is not None else 0

        # 4. Calculate similarity
        doc_count = len(doc_keypoints)
        blank_count = blank_features.feature_count
        similarity_score = inlier_count / min(doc_count, blank_count) if min(doc_count, blank_count) > 0 else 0.0

        # 5. Determine auxiliary_has_content
        auxiliary_has_content = similarity_score < similarity_threshold

        elapsed = (time.time() - start_time) * 1000

        return ROIComparisonResult(
            field_id=field_id,
            similarity_score=similarity_score,
            inlier_count=inlier_count,
            doc_feature_count=doc_count,
            blank_feature_count=blank_count,
            auxiliary_has_content=auxiliary_has_content,
            comparison_time_ms=elapsed
        )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return ROIComparisonResult(
            field_id=field_id,
            similarity_score=0.0,
            inlier_count=0,
            doc_feature_count=0,
            blank_feature_count=blank_features.feature_count,
            auxiliary_has_content=None,
            comparison_time_ms=elapsed,
            error_message=str(e)
        )
```

**Unit Tests** (`tests/unit/test_roi_comparator.py`):
- test_compare_identical_rois()
- test_compare_filled_vs_blank()
- test_compare_empty_vs_blank()
- test_insufficient_features()
- test_threshold_boundary()

---

### Phase 4: VLM Recognizer Integration

**Goal**: Integrate auxiliary comparison into VLMRecognizer

**Files to Modify**:
- `vlm_pdf_recognizer/recognition/vlm_recognizer.py`

**Implementation Steps**:

1. **Extend RecognitionResult dataclass**:
```python
@dataclass
class RecognitionResult:
    # ... existing fields ...

    # NEW fields
    auxiliary_has_content: Optional[bool] = None
    auxiliary_similarity_score: Optional[float] = None
    auxiliary_comparison_time_ms: Optional[float] = None
```

2. **Modify _recognize_field() to add auxiliary comparison**:
```python
def _recognize_field(
    self,
    roi_image: np.ndarray,
    field_schema: FieldSchema,
    blank_roi_cache: 'BlankROIFeatureCache' = None,  # NEW parameter
    template_id: str = None  # NEW parameter
) -> RecognitionResult:
    # Skip auxiliary for title and checkbox fields
    auxiliary_result = None
    if field_schema.field_type not in ("title", "checkbox") and blank_roi_cache:
        blank_features = blank_roi_cache.get_features(template_id, field_schema.field_id)
        if blank_features:
            from vlm_pdf_recognizer.alignment.roi_comparator import compare_roi_to_blank
            auxiliary_result = compare_roi_to_blank(roi_image, blank_features, field_id=field_schema.field_id)

    # Conditional VLM execution
    skip_vlm = (auxiliary_result and not auxiliary_result.auxiliary_has_content and not auxiliary_result.error_message)

    if skip_vlm:
        # Empty field, skip VLM
        has_content = None
        content_text = None
    else:
        # Run VLM inference (existing code)
        # ...

    return RecognitionResult(
        field_id=field_schema.field_id,
        has_content=has_content,
        content_text=content_text,
        auxiliary_has_content=auxiliary_result.auxiliary_has_content if auxiliary_result else None,
        auxiliary_similarity_score=auxiliary_result.similarity_score if auxiliary_result else None,
        # ... other fields ...
    )
```

3. **Update process_document() to pass blank_roi_cache**:
```python
def process_document(
    self,
    roi_images: List[np.ndarray],
    template_id: str,
    page_number: int,
    document_name: str,
    blank_roi_cache: 'BlankROIFeatureCache' = None  # NEW parameter
) -> DocumentRecognitionOutput:
    # ... existing code ...

    for roi_image, field_schema in zip(roi_images, template_schema.field_schemas):
        result = self._recognize_field(
            roi_image,
            field_schema,
            blank_roi_cache=blank_roi_cache,  # NEW
            template_id=template_id  # NEW
        )
        field_results.append(result)

    # ... rest of existing code ...
```

4. **Update calculate_results_status() to use auxiliary_has_content**:
```python
def calculate_results_status(self) -> bool:
    # Use auxiliary_has_content with fallback to VLM has_content
    # Implementation as per data-model.md
    pass
```

**Unit Tests** (`tests/unit/test_vlm_recognizer.py`):
- Modify existing tests to handle auxiliary fields
- Test auxiliary comparison integration
- Test VLM skip on empty fields
- Test VLM run on filled fields

---

### Phase 5: Output Schema Extension

**Goal**: Add auxiliary columns to outputs

**Files to Modify**:
- `vlm_pdf_recognizer/output.py`

**Implementation Steps**:

1. **Modify DocumentRecognitionOutput.to_json_dict()**:
```python
def to_json_dict(self) -> Dict[str, Any]:
    # ... existing code ...

    for result in self.field_results:
        if result.has_content is None:  # Title field
            continue

        field_data = {
            "has_content": result.has_content,
            "content_text": result.content_text,
            "auxiliary_has_content": result.auxiliary_has_content,  # NEW
            "auxiliary_similarity_score": result.auxiliary_similarity_score  # NEW
        }

        fields[result.field_id] = field_data

    # ... rest of existing code ...
```

2. **Modify save_vlm_visualization() to use auxiliary results**:
```python
def save_vlm_visualization(result: ProcessingResult, vlm_output, output_dir: str):
    # ... existing code to create visualization ...

    for roi in result.extracted_rois:
        field_result = field_results_dict.get(roi.roi_id)

        # Use auxiliary_has_content for coloring (with fallback to VLM has_content)
        has_content = field_result.auxiliary_has_content if field_result.auxiliary_has_content is not None else field_result.has_content

        if field_result.has_content is None:  # Title
            color = roi.visualization_color
        elif not has_content:  # Empty (auxiliary or VLM)
            color = (0, 0, 255)  # Red
        else:  # Filled (auxiliary or VLM)
            color = roi.visualization_color  # Green

        # ... draw boxes ...
```

**Integration Tests**:
- Test JSON output contains auxiliary fields
- Test visualization uses auxiliary coloring
- Test backward compatibility (missing auxiliary data)

---

### Phase 6: Main Pipeline Integration

**Goal**: Pass blank_roi_cache through the pipeline

**Files to Modify**:
- `main.py`

**Implementation Steps**:

1. **Modify VLM recognition section**:
```python
# Around line 150 in main.py
if args.enable_vlm and vlm_recognizer and result.extracted_rois:
    # ... existing code ...

    vlm_output = vlm_recognizer.process_document(
        roi_images=roi_images,
        template_id=result.matched_template_id,
        page_number=result.page_number,
        document_name=Path(result.input_path).name,
        blank_roi_cache=processor.blank_roi_cache  # NEW
    )

    # ... rest of existing code ...
```

**Integration Tests** (`tests/integration/test_vlm_pipeline.py`):
- test_end_to_end_with_auxiliary()
- test_backward_compat_missing_features()
- test_batch_processing_with_auxiliary()

---

## Testing Strategy

### Test Execution Order

1. **Unit Tests**: Run after each phase
   ```bash
   pytest tests/unit/test_blank_roi_cache.py -v
   pytest tests/unit/test_roi_comparator.py -v
   pytest tests/unit/test_vlm_recognizer.py -v
   ```

2. **Integration Tests**: Run after Phase 6
   ```bash
   pytest tests/integration/test_vlm_pipeline.py -v
   ```

3. **Backward Compatibility Tests**: Run after all phases
   ```bash
   pytest tests/integration/test_backward_compat.py -v
   ```

4. **Full Test Suite**: Final validation
   ```bash
   pytest tests/ -v --cov=vlm_pdf_recognizer
   ```

### Test Data Requirements

- **Blank templates**: Use existing templates/images/{template_id}.png
- **Filled documents**: Use existing test documents in tests/fixtures/
- **Partially filled**: Create new test documents with mix of empty/filled fields
- **Edge cases**: Very faint signatures, light stamps

---

## Validation Checklist

Before marking implementation complete:

- [ ] Configuration script generates blank ROI features for all templates
- [ ] BlankROIFeatureCache loads features without errors
- [ ] Auxiliary comparison runs on non-title, non-checkbox fields
- [ ] VLM inference is skipped for empty fields (auxiliary_has_content=False)
- [ ] VLM inference runs for filled fields (auxiliary_has_content=True)
- [ ] Outputs include auxiliary columns (JSON and CSV)
- [ ] Visualization uses auxiliary-based coloring
- [ ] Validation logic uses auxiliary results
- [ ] Missing blank features gracefully fall back to VLM-only
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Backward compatibility verified
- [ ] Performance meets targets (<50ms per ROI)
- [ ] Memory footprint acceptable (<20MB for cache)

---

## Common Issues and Troubleshooting

### Issue: Missing blank_roi_features.npz

**Symptom**: Warning logs "Blank features not found for template {template_id}"

**Solution**: Run `python update_configs.py` to generate blank features

### Issue: Feature extraction fails on ROI

**Symptom**: Error "Document ROI has <4 keypoints"

**Solution**: ROI is too uniform (e.g., solid white area). This is expected behavior - auxiliary comparison will return None and fall back to VLM.

### Issue: Similarity scores always high (all fields marked empty)

**Symptom**: All fields have similarity_score > 0.6

**Solution**: Check that blank template image doesn't contain filled content. Blank template should be completely unfilled.

### Issue: Performance slower than expected

**Symptom**: Auxiliary comparison takes >100ms per ROI

**Solution**: Check if SIFT parameters are correct. Ensure using `is_template=False` for document ROIs (uses faster parameters).

---

## Next Steps

After successful implementation:

1. Run full test suite: `pytest tests/ -v --cov`
2. Test with real document samples
3. Tune similarity threshold if needed (adjust 0.6 value)
4. Monitor accuracy metrics and performance
5. Document any threshold adjustments in research.md

---

## Support

For implementation questions or issues:
1. Review [contracts/](contracts/) for API details
2. Check [data-model.md](data-model.md) for entity definitions
3. Refer to existing template matching code for SIFT usage examples
4. Review [research.md](research.md) for design rationale
