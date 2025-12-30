# ROI Comparator API Contract

**Module**: `vlm_pdf_recognizer.alignment.roi_comparator`
**Purpose**: Compare document ROIs against blank template ROIs using SIFT feature matching

## Public Functions

### compare_roi_to_blank()

Compare a document ROI image against blank template ROI features.

**Signature**:
```python
def compare_roi_to_blank(
    doc_roi_image: np.ndarray,
    blank_features: BlankROIFeatures,
    similarity_threshold: float = 0.6,
    field_id: str = None
) -> ROIComparisonResult
```

**Parameters**:
- `doc_roi_image` (np.ndarray): Document ROI image, BGR format, uint8, shape (H, W, 3)
- `blank_features` (BlankROIFeatures): Blank template ROI features (keypoints, descriptors)
- `similarity_threshold` (float, optional): Threshold for filled/unfilled determination, default 0.6
- `field_id` (str, optional): Field identifier for logging, default None

**Returns**:
- `ROIComparisonResult`: Comparison result with similarity score and auxiliary_has_content determination

**Behavior**:
1. Extract SIFT features from doc_roi_image using feature_extractor.extract_features()
2. Match doc features against blank_features.descriptors using FLANN matcher
3. Compute RANSAC homography and inlier count
4. Calculate similarity_score = inlier_count / min(doc_feature_count, blank_feature_count)
5. Determine auxiliary_has_content:
   - False if similarity_score >= similarity_threshold (high similarity, unfilled)
   - True if similarity_score < similarity_threshold (low similarity, potential content)

**Error Handling**:
- If doc_roi_image has insufficient features (<4 keypoints): Return ROIComparisonResult with error_message, auxiliary_has_content=None
- If FLANN matching fails: Return ROIComparisonResult with error_message, auxiliary_has_content=None
- If RANSAC fails (no homography found): Return ROIComparisonResult with similarity_score=0.0, auxiliary_has_content=True (assume filled)

**Performance**:
- Expected time: 30-50ms per ROI on standard hardware
- Timeout: If exceeds 500ms, log warning but complete operation

**Example**:
```python
from vlm_pdf_recognizer.alignment.roi_comparator import compare_roi_to_blank

result = compare_roi_to_blank(
    doc_roi_image=roi_image,
    blank_features=blank_cache.get_features("contractor_1", "person1"),
    similarity_threshold=0.6,
    field_id="person1"
)

if result.error_message:
    logger.warning(f"Comparison failed: {result.error_message}")
    # Fall back to VLM inference
elif result.auxiliary_has_content:
    logger.info(f"Field {field_id} appears filled (similarity={result.similarity_score:.2f})")
    # Proceed with VLM inference
else:
    logger.info(f"Field {field_id} appears empty (similarity={result.similarity_score:.2f})")
    # Skip VLM inference
```

---

## Data Classes

### ROIComparisonResult

**Attributes**:
```python
@dataclass
class ROIComparisonResult:
    field_id: str
    similarity_score: float  # Range [0.0, 1.0]
    inlier_count: int
    doc_feature_count: int
    blank_feature_count: int
    auxiliary_has_content: Optional[bool]  # True=filled, False=empty, None=error
    comparison_time_ms: float
    error_message: Optional[str] = None
```

**Validation**:
- similarity_score must be in [0.0, 1.0]
- inlier_count must be <= min(doc_feature_count, blank_feature_count)
- If error_message is not None, auxiliary_has_content should be None

### BlankROIFeatures

**Attributes**:
```python
@dataclass
class BlankROIFeatures:
    field_id: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray  # Shape (N, 128), dtype float32
    feature_count: int  # Derived from len(keypoints)
```

**Validation**:
- len(keypoints) == descriptors.shape[0]
- descriptors.shape[1] == 128 (SIFT descriptor length)
- feature_count > 0

---

## Error Codes and Messages

| Error Code | Message | Cause | Recovery |
|------------|---------|-------|----------|
| INSUFFICIENT_DOC_FEATURES | "Document ROI has <4 keypoints, cannot perform matching" | ROI is too uniform or small | Set auxiliary_has_content=None, proceed with VLM |
| INSUFFICIENT_BLANK_FEATURES | "Blank template has <4 keypoints, invalid blank features" | Blank template extraction error | Set auxiliary_has_content=None, proceed with VLM |
| MATCHING_FAILED | "FLANN matcher raised exception: {exception}" | OpenCV FLANN error | Set auxiliary_has_content=None, proceed with VLM |
| RANSAC_FAILED | "RANSAC could not find homography, no inliers" | Features too dissimilar | Set similarity_score=0.0, auxiliary_has_content=True (assume filled) |
| FEATURE_EXTRACTION_FAILED | "SIFT feature extraction failed: {exception}" | OpenCV error or corrupted image | Set auxiliary_has_content=None, proceed with VLM |

---

## Integration Points

### Caller: VLMRecognizer._recognize_field()

**Usage**:
```python
# In vlm_pdf_recognizer/recognition/vlm_recognizer.py

def _recognize_field(self, roi_image, field_schema, blank_features_cache):
    # Skip auxiliary for title and checkbox fields
    if field_schema.field_type in ("title", "checkbox"):
        auxiliary_result = None  # Skipped
    else:
        # Get blank features from cache
        blank_features = blank_features_cache.get_features(
            self.template_id,
            field_schema.field_id
        )

        if blank_features is None:
            # Blank features not available, skip auxiliary
            auxiliary_result = None
        else:
            # Perform auxiliary comparison
            auxiliary_result = compare_roi_to_blank(
                doc_roi_image=roi_image,
                blank_features=blank_features,
                similarity_threshold=0.6,
                field_id=field_schema.field_id
            )

    # Determine whether to run VLM
    if auxiliary_result is None or auxiliary_result.auxiliary_has_content:
        # Run VLM inference
        vlm_result = self._call_vlm(roi_image, prompt)
    else:
        # Skip VLM (field is empty)
        vlm_result = None

    # Create RecognitionResult with both auxiliary and VLM data
    return RecognitionResult(
        field_id=field_schema.field_id,
        auxiliary_has_content=auxiliary_result.auxiliary_has_content if auxiliary_result else None,
        auxiliary_similarity_score=auxiliary_result.similarity_score if auxiliary_result else None,
        has_content=vlm_result.has_content if vlm_result else None,
        content_text=vlm_result.content_text if vlm_result else None,
        ...
    )
```

### Dependency: feature_extractor.extract_features()

**Existing Contract**:
```python
def extract_features(
    image: np.ndarray,
    is_template: bool = False
) -> Tuple[List[cv2.KeyPoint], np.ndarray]
```

**Usage in ROI Comparator**:
- Call with `is_template=False` for document ROIs
- Returns (keypoints, descriptors) tuple
- Handles grayscale conversion internally

### Dependency: template_matcher.match_features() and compute_homography_and_inliers()

**Existing Contracts**:
```python
def match_features(
    doc_descriptors: np.ndarray,
    template_descriptors: np.ndarray,
    ratio_threshold: float = 0.7
) -> List[cv2.DMatch]

def compute_homography_and_inliers(
    doc_keypoints: List[cv2.KeyPoint],
    template_keypoints: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    ransac_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]
```

**Usage in ROI Comparator**:
- Reuse FLANN-based matching with Lowe's ratio test
- Reuse RANSAC homography estimation
- Same threshold values as template matching (ratio=0.7, RANSAC=5.0)

---

## Testing Requirements

### Unit Tests

**Test File**: `tests/unit/test_roi_comparator.py`

**Test Cases**:
1. **test_compare_identical_rois**: Compare ROI to itself → similarity ≈ 1.0, auxiliary_has_content=False
2. **test_compare_filled_vs_blank**: Compare filled ROI to blank → similarity < 0.6, auxiliary_has_content=True
3. **test_compare_empty_vs_blank**: Compare empty ROI to blank → similarity >= 0.6, auxiliary_has_content=False
4. **test_insufficient_features**: ROI with <4 keypoints → error_message, auxiliary_has_content=None
5. **test_threshold_boundary**: similarity_score exactly 0.6 → auxiliary_has_content=False (>= threshold)
6. **test_dimension_mismatch**: Different sized ROIs → handles gracefully, returns valid result
7. **test_feature_extraction_error**: Corrupted image → error_message, auxiliary_has_content=None
8. **test_performance**: Comparison completes in <100ms for typical ROI

### Integration Tests

**Test File**: `tests/integration/test_vlm_pipeline.py`

**Test Cases**:
1. **test_end_to_end_with_auxiliary**: Process document with auxiliary enabled → correct auxiliary_has_content values
2. **test_vlm_skip_on_empty_fields**: Empty fields → VLM inference skipped, has_content=None
3. **test_vlm_run_on_filled_fields**: Filled fields → VLM inference runs, has_content populated

---

## Performance Contract

| Metric | Target | Measurement |
|--------|--------|-------------|
| Per-ROI comparison time | <50ms (median) | Time from compare_roi_to_blank() call to return |
| Per-ROI comparison time | <100ms (p95) | 95th percentile across all ROIs |
| Memory usage | <100KB per comparison | Peak memory delta during single comparison |
| Accuracy (filled detection) | >95% | True positive rate on test set with clearly filled fields |
| Accuracy (empty detection) | >95% | True negative rate on test set with clearly empty fields |

---

## Backward Compatibility

This is a new module with no backward compatibility concerns. Integration with existing modules (feature_extractor, template_matcher) uses stable public APIs that have been in production since feature 001.

---

## Future Extensions

Potential enhancements (out of scope for this feature):
- Per-field-type thresholds (different threshold for stamp vs text vs signature)
- Multi-template blank feature averaging (improve robustness to printing variations)
- Adaptive threshold tuning based on historical accuracy
- Sub-ROI analysis (detect partial filling within ROI)
