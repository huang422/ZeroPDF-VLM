# Blank ROI Feature Cache API Contract

**Module**: `vlm_pdf_recognizer.alignment.blank_roi_cache`
**Purpose**: Load and cache blank template ROI features for runtime auxiliary comparison

## Public Classes

### BlankROIFeatureCache

In-memory cache of blank ROI features for all templates.

**Constructor**:
```python
class BlankROIFeatureCache:
    def __init__(self):
        """Initialize empty cache."""
        self.templates: Dict[str, Dict[str, BlankROIFeatures]] = {}
        self.loaded_templates: Set[str] = set()
        self.failed_templates: Set[str] = set()
```

---

## Public Methods

### load_from_directory()

Load all blank ROI features from template data directory.

**Signature**:
```python
def load_from_directory(self, templates_dir: str) -> None
```

**Parameters**:
- `templates_dir` (str): Path to templates data directory (e.g., "data")

**Behavior**:
1. Scan for all subdirectories in templates_dir matching template_id pattern
2. For each template directory:
   a. Check for blank_roi_features.npz file
   b. If found: Load features using numpy.load()
   c. Parse field_id keys and reconstruct BlankROIFeatures objects
   d. Add to self.templates[template_id]
   e. Mark template as loaded in self.loaded_templates
3. If .npz file missing or corrupted:
   a. Log warning message
   b. Add template to self.failed_templates
   c. Continue processing other templates (don't fail entire load)

**Error Handling**:
- Missing .npz file: Log `WARNING: Blank features not found for template {template_id}`, add to failed_templates
- Corrupted .npz file: Log `ERROR: Failed to load blank features for {template_id}: {exception}`, add to failed_templates
- Invalid feature data (wrong shape): Log `ERROR: Invalid blank feature data for {template_id}.{field_id}`, skip that field

**Performance**:
- Expected time: <200ms for loading all 3 templates (~29 fields total)
- Memory: ~15MB for all loaded features

**Example**:
```python
cache = BlankROIFeatureCache()
cache.load_from_directory("data")

print(f"Loaded templates: {cache.loaded_templates}")
# Output: {'contractor_1', 'contractor_2', 'enterprise_1'}

print(f"Failed templates: {cache.failed_templates}")
# Output: set() (or list of any failed loads)
```

---

### get_features()

Retrieve blank ROI features for a specific template and field.

**Signature**:
```python
def get_features(
    self,
    template_id: str,
    field_id: str
) -> Optional[BlankROIFeatures]
```

**Parameters**:
- `template_id` (str): Template identifier (e.g., "contractor_1")
- `field_id` (str): Field identifier (e.g., "person1", "big")

**Returns**:
- `BlankROIFeatures` if found
- `None` if template or field not found

**Behavior**:
1. Check if template_id exists in self.templates
2. If not found: Return None
3. Check if field_id exists in self.templates[template_id]
4. If not found: Return None
5. Return BlankROIFeatures object

**Error Handling**:
- Template not loaded: Return None (caller handles fallback to VLM)
- Field not found: Return None (caller handles fallback to VLM)
- No exceptions raised

**Performance**: O(1) dictionary lookup, <1ms

**Example**:
```python
features = cache.get_features("contractor_1", "person1")

if features is None:
    logger.warning("Blank features not available, falling back to VLM-only")
    # Proceed with VLM inference
else:
    logger.info(f"Using blank features ({features.feature_count} features)")
    # Perform auxiliary comparison
```

---

### has_features()

Check if a template has blank features available.

**Signature**:
```python
def has_features(self, template_id: str) -> bool
```

**Parameters**:
- `template_id` (str): Template identifier

**Returns**:
- `True` if template was successfully loaded
- `False` if template failed to load or was not found

**Behavior**:
- Return `template_id in self.loaded_templates`

**Performance**: O(1) set lookup, <1ms

**Example**:
```python
if cache.has_features("contractor_1"):
    logger.info("Auxiliary comparison available for contractor_1")
else:
    logger.warning("Auxiliary comparison unavailable for contractor_1, using VLM-only mode")
```

---

### get_loaded_count()

Get count of successfully loaded templates.

**Signature**:
```python
def get_loaded_count(self) -> int
```

**Returns**:
- Number of templates with successfully loaded blank features

**Example**:
```python
count = cache.get_loaded_count()
logger.info(f"Loaded blank features for {count} templates")
```

---

## Data Structures

### BlankROIFeatures

**Dataclass**:
```python
@dataclass
class BlankROIFeatures:
    field_id: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray  # Shape (N, 128), dtype float32
    feature_count: int  # Derived: len(keypoints)

    def __post_init__(self):
        """Validate feature data consistency."""
        assert len(self.keypoints) == self.descriptors.shape[0], \
            "Keypoints and descriptors count mismatch"
        assert self.descriptors.shape[1] == 128, \
            "Invalid SIFT descriptor length"
        assert self.feature_count == len(self.keypoints), \
            "Feature count mismatch"
```

---

## File Format

### blank_roi_features.npz

NumPy compressed archive containing SIFT features for all fields in a template.

**File Location**: `data/{template_id}/blank_roi_features.npz`

**Structure**:
```python
# Created with numpy.savez():
np.savez(
    "data/contractor_1/blank_roi_features.npz",
    person1_keypoints=keypoints_array,
    person1_descriptors=descriptors_array,
    company1_keypoints=keypoints_array,
    company1_descriptors=descriptors_array,
    # ... one pair per field_id
)

# Loaded with numpy.load():
npz_file = np.load("data/contractor_1/blank_roi_features.npz")
person1_kp = npz_file['person1_keypoints']
person1_desc = npz_file['person1_descriptors']
```

**Key Naming Convention**:
- Keypoints: `{field_id}_keypoints`
- Descriptors: `{field_id}_descriptors`

**Data Types**:
- Keypoints: Structured array with fields (x, y, size, angle, response, octave, class_id)
- Descriptors: float32 array, shape (N, 128)

**File Size**: ~100-500KB per template (depends on number of fields and feature density)

---

## Integration Points

### Caller: DocumentProcessor.__init__()

**Usage**:
```python
# In vlm_pdf_recognizer/pipeline.py

class DocumentProcessor:
    def __init__(self, templates_dir: str = "data", verbose: bool = False):
        self.templates_dir = templates_dir
        self.verbose = verbose

        # Load golden templates (existing)
        self.templates = load_all_templates(templates_dir)

        # Load blank ROI features cache (NEW)
        self.blank_roi_cache = BlankROIFeatureCache()
        self.blank_roi_cache.load_from_directory(templates_dir)

        if self.verbose:
            loaded = len(self.blank_roi_cache.loaded_templates)
            failed = len(self.blank_roi_cache.failed_templates)
            print(f"Loaded blank ROI features: {loaded} templates")
            if failed > 0:
                print(f"  Warning: {failed} templates failed to load (using VLM-only mode)")
```

### Caller: VLMRecognizer.process_document()

**Usage**:
```python
# In vlm_pdf_recognizer/recognition/vlm_recognizer.py

def process_document(
    self,
    roi_images: List[np.ndarray],
    template_id: str,
    blank_roi_cache: BlankROIFeatureCache  # NEW parameter
):
    # Check if auxiliary comparison available for this template
    has_auxiliary = blank_roi_cache.has_features(template_id)

    for roi_image, field_schema in zip(roi_images, template_schema.field_schemas):
        if has_auxiliary and field_schema.field_type not in ("title", "checkbox"):
            # Get blank features for this field
            blank_features = blank_roi_cache.get_features(template_id, field_schema.field_id)
            # Perform auxiliary comparison
            # ...
        else:
            # Skip auxiliary, use VLM-only
            # ...
```

---

## Error Handling Contract

| Error Scenario | Behavior | User Impact |
|----------------|----------|-------------|
| Missing .npz file | Log warning, add to failed_templates, continue | Template uses VLM-only mode, no auxiliary comparison |
| Corrupted .npz file | Log error, add to failed_templates, continue | Template uses VLM-only mode, no auxiliary comparison |
| Invalid feature data | Log error, skip that field, continue | Field uses VLM-only mode for that field |
| Empty templates_dir | Load completes with empty cache | All templates use VLM-only mode |
| Permission denied | Raise IOError (propagate to caller) | Initialization fails, user must fix permissions |

---

## Testing Requirements

### Unit Tests

**Test File**: `tests/unit/test_blank_roi_cache.py`

**Test Cases**:
1. **test_load_valid_features**: Load from directory with valid .npz files → templates loaded successfully
2. **test_load_missing_file**: Template directory without .npz → added to failed_templates, no crash
3. **test_load_corrupted_file**: Corrupted .npz file → added to failed_templates, other templates load OK
4. **test_get_features_exists**: get_features() for loaded field → returns BlankROIFeatures
5. **test_get_features_missing_template**: get_features() for unloaded template → returns None
6. **test_get_features_missing_field**: get_features() for non-existent field → returns None
7. **test_has_features_loaded**: has_features() for loaded template → returns True
8. **test_has_features_failed**: has_features() for failed template → returns False
9. **test_empty_cache**: No templates directory → empty cache, no errors
10. **test_feature_validation**: Invalid descriptor shape → logs error, skips field

### Integration Tests

**Test File**: `tests/integration/test_vlm_pipeline.py`

**Test Cases**:
1. **test_processor_init_with_blank_features**: DocumentProcessor loads cache → blank_roi_cache available
2. **test_pipeline_with_missing_features**: Process document with failed template → falls back to VLM-only
3. **test_memory_footprint**: Load all templates → memory usage <20MB

---

## Performance Contract

| Metric | Target | Measurement |
|--------|--------|-------------|
| Load time (all templates) | <200ms | Time from load_from_directory() call to return |
| Memory footprint | <20MB | Total memory used by cache after loading all templates |
| get_features() latency | <1ms | Time for single lookup |
| Cache initialization | <500ms | Total time to load cache in DocumentProcessor.__init__() |

---

## Backward Compatibility

This is a new module with no backward compatibility concerns. If the blank_roi_features.npz file is missing, the system gracefully falls back to existing VLM-only behavior.

---

## Future Extensions

Potential enhancements (out of scope for this feature):
- Lazy loading: Load features on-demand per template
- Feature versioning: Support multiple versions of blank features
- Hot-reload: Detect .npz file changes and reload cache
- Compression: Use more aggressive NumPy compression for large feature sets
