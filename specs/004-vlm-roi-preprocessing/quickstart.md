# Quickstart Guide: ROI Preprocessing Pipeline

**Feature**: 004-vlm-roi-preprocessing
**Created**: 2025-12-31
**Target Audience**: Developers implementing the preprocessing feature

## Overview

This guide provides step-by-step instructions for implementing the ROI preprocessing pipeline. Follow these steps in order to build the feature incrementally with continuous testing.

---

## Prerequisites

Before starting implementation:

1. **Read Design Documents**:
   - [spec.md](spec.md) - Feature requirements and user scenarios
   - [plan.md](plan.md) - Technical architecture and structure
   - [research.md](research.md) - Technical decisions and rationale
   - [data-model.md](data-model.md) - Data structures and relationships

2. **Set Up Environment**:
   ```bash
   # Ensure vlmcv environment is activated
   conda activate vlmcv

   # Verify dependencies (should already be installed)
   python -c "import cv2, numpy; print('OpenCV:', cv2.__version__)"
   ```

3. **Understand Existing Codebase**:
   - Review `vlm_pdf_recognizer/recognition/vlm_recognizer.py` (integration point)
   - Review `vlm_pdf_recognizer/alignment/roi_comparator.py` (existing pixel-based detection patterns)
   - Review `vlm_pdf_recognizer/recognition/config.py` (configuration pattern)
   - Review `update_configs.py` (template initialization pattern)

---

## Implementation Phases

### Phase 1: Configuration Setup (30 minutes)

**Goal**: Add preprocessing configuration parameters to config.py

**Steps**:

1. Open `vlm_pdf_recognizer/recognition/config.py`

2. Add preprocessing configuration section after existing checkbox parameters:

```python
# ==============================================================================
# ROI PREPROCESSING PIPELINE PARAMETERS (Feature 004-vlm-roi-preprocessing)
# ==============================================================================

# Template Difference
DIFFERENCE_THRESHOLD = 25
USE_CLAHE_NORMALIZATION = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
MEDIAN_FILTER_SIZE = 3

# HSV Saturation
SATURATION_MIN_THRESHOLD = 15
SATURATION_BOOST_FACTOR = 2.0
SATURATION_VALUE_BLEND_RATIO = 0.7

# Morphological Operations
MORPHOLOGY_LINE_RATIO = 0.3
MIN_SIGNATURE_AREA = 500
USE_MULTI_SCALE_MORPHOLOGY = False

# Connected Components
MIN_BLOB_AREA = 20
COMPONENT_COUNT_THRESHOLD = 3
MAX_ASPECT_RATIO = 10
MIN_COMPONENT_DENSITY = 0.3
INK_RATIO_THRESHOLD = 0.005
USE_8_CONNECTIVITY = True

# Performance
DEBUG_SAVE_INTERMEDIATE_IMAGES = False
PNG_COMPRESSION_LEVEL = 6
PRE_ALLOCATE_WORK_BUFFERS = True
MAX_ROI_DIMENSION = 1000
```

**Test**: Verify config loads without errors:
```bash
python -c "from vlm_pdf_recognizer.recognition import config; print('Config OK')"
```

---

### Phase 2: Blank Template ROI Extraction (1 hour)

**Goal**: Extend update_configs.py to generate blank template ROIs

**Steps**:

1. Open `update_configs.py`

2. Add function to extract blank ROIs (after `extract_blank_roi_features` function):

```python
def extract_and_save_blank_rois(
    template_id: str,
    template_image: np.ndarray,
    rois: list
) -> None:
    """Extract and save blank template ROI images."""
    from vlm_pdf_recognizer.extraction.roi_extractor import extract_rois

    # Create blank_rois directory
    output_dir = Path("data") / template_id / "blank_rois"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Extracting blank ROIs for {template_id}...")

    # Extract ROIs using existing roi_extractor logic
    roi_images = extract_rois(template_image, [roi['coordinates'] for roi in rois])

    # Save each ROI as PNG
    for roi_config, roi_image in zip(rois, roi_images):
        field_id = roi_config['id']
        output_path = output_dir / f"{field_id}.png"
        cv2.imwrite(str(output_path), roi_image)

    print(f"   → Generated {len(roi_images)} blank ROI images")
```

3. Integrate into main() function (after existing feature extraction):

```python
def main():
    # ... existing code ...

    for template_id in template_ids:
        # ... existing extraction code ...

        # NEW: Extract and save blank ROIs
        extract_and_save_blank_rois(template_id, template_image, rois)

        print(f"✓ Template '{template_id}': Configuration complete")
```

**Test**: Run update_configs.py and verify blank ROI files are created:
```bash
python update_configs.py

# Verify output
ls -la data/contractor_1/blank_rois/
# Should see title.png, VX1.png, VX2.png, year.png, month.png, etc.
```

**Validation**: Check blank ROI images visually:
```bash
# Open with image viewer
eog data/contractor_1/blank_rois/signature.png
# Should show blank signature box from template
```

---

### Phase 3: ROI Preprocessor Core Implementation (3-4 hours)

**Goal**: Implement roi_preprocessor.py with 6-step pipeline

**Steps**:

1. Create `vlm_pdf_recognizer/recognition/roi_preprocessor.py`:

```python
"""ROI Preprocessing Pipeline for Content Detection."""

import time
import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
from vlm_pdf_recognizer.recognition import config

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline for a single ROI."""
    field_id: str
    has_content: bool | None
    ink_ratio: float | None
    component_count: int | None
    processing_time_ms: float
    error_message: str | None = None


class ROIPreprocessor:
    """6-step preprocessing pipeline for ROI content detection."""

    def __init__(self, save_debug_images: bool = False, output_dir: Path | None = None):
        self.save_debug_images = save_debug_images
        self.output_dir = Path(output_dir) if output_dir else None
        self.horizontal_kernels = {}
        self.vertical_kernels = {}
        self.noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        if save_debug_images and not output_dir:
            raise ValueError("output_dir required when save_debug_images=True")

    def preprocess_roi(
        self,
        doc_roi_image: np.ndarray,
        blank_template_roi: np.ndarray,
        field_id: str,
        document_name: str | None = None
    ) -> PreprocessingResult:
        """Execute 6-step preprocessing pipeline."""
        start_time = time.time()

        try:
            # Resize if needed
            if doc_roi_image.shape != blank_template_roi.shape:
                logger.warning(f"ROI size mismatch for {field_id}, resizing...")
                doc_roi_image = cv2.resize(
                    doc_roi_image,
                    (blank_template_roi.shape[1], blank_template_roi.shape[0])
                )

            intermediate_images = {}

            # Step 1: HSV Saturation Extraction
            saturation = self._extract_saturation_channel(doc_roi_image)
            intermediate_images['01_saturation'] = saturation

            template_saturation = self._extract_saturation_channel(blank_template_roi)

            # Step 2: Template Difference
            difference = self._compute_template_difference(saturation, template_saturation)
            intermediate_images['02_difference'] = difference

            # Step 3-4: Line Removal
            lines_removed = self._remove_form_lines(difference)
            intermediate_images['03_lines_removed'] = lines_removed

            # Step 5: Noise Removal
            cleaned = self._remove_noise(lines_removed)
            intermediate_images['04_noise_removed'] = cleaned

            # Step 6: Feature Extraction & Decision
            component_count, ink_ratio = self._extract_features(cleaned)
            has_content, reasoning = self._determine_has_content(component_count, ink_ratio)

            intermediate_images['05_final_binary'] = cleaned

            # Save debug images if enabled
            if self.save_debug_images and document_name:
                self._save_intermediate_images(
                    intermediate_images, field_id, document_name
                )
                self._save_metadata(
                    field_id, document_name, has_content, ink_ratio,
                    component_count, reasoning
                )

            processing_time_ms = (time.time() - start_time) * 1000

            logger.debug(f"Field {field_id}: {reasoning}, time={processing_time_ms:.1f}ms")

            return PreprocessingResult(
                field_id=field_id,
                has_content=has_content,
                ink_ratio=ink_ratio,
                component_count=component_count,
                processing_time_ms=processing_time_ms,
                error_message=None
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Preprocessing failed for {field_id}: {e}")
            return PreprocessingResult(
                field_id=field_id,
                has_content=None,
                ink_ratio=None,
                component_count=None,
                processing_time_ms=processing_time_ms,
                error_message=str(e)
            )

    # Implement internal methods: _extract_saturation_channel, etc.
    # (See research.md for detailed implementation)
```

2. Implement the 6 internal methods (copy from research.md examples)

**Test**: Create unit test `tests/unit/test_roi_preprocessor.py`:

```python
import numpy as np
import pytest
from vlm_pdf_recognizer.recognition.roi_preprocessor import ROIPreprocessor

def test_preprocessor_detects_blank_roi():
    """Test that blank ROI is detected as has_content=False."""
    preprocessor = ROIPreprocessor()

    # Create blank ROI (white image)
    blank_roi = np.ones((100, 200, 3), dtype=np.uint8) * 255

    result = preprocessor.preprocess_roi(blank_roi, blank_roi, "test_field")

    assert result.has_content is False
    assert result.ink_ratio < 0.005
    assert result.component_count < 3

def test_preprocessor_detects_signature():
    """Test that signature ROI is detected as has_content=True."""
    preprocessor = ROIPreprocessor()

    # Create blank template
    blank_roi = np.ones((100, 200, 3), dtype=np.uint8) * 255

    # Create signature ROI (add black signature)
    signature_roi = blank_roi.copy()
    cv2.rectangle(signature_roi, (50, 30), (150, 70), (0, 0, 0), -1)

    result = preprocessor.preprocess_roi(signature_roi, blank_roi, "signature")

    assert result.has_content is True
    assert result.ink_ratio > 0.005
    assert result.component_count >= 1
```

Run test:
```bash
pytest tests/unit/test_roi_preprocessor.py -v
```

---

### Phase 4: Blank Template ROI Cache (1 hour)

**Goal**: Create cache to load blank ROIs at runtime

**Steps**:

1. Create `vlm_pdf_recognizer/alignment/blank_template_roi_cache.py`:

```python
"""Cache for blank template ROI images."""

import logging
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BlankTemplateROICache:
    """Runtime cache for blank template ROI images."""

    def __init__(self, templates_dir: str | Path):
        self.templates_dir = Path(templates_dir)
        self.cache: dict[str, dict[str, np.ndarray]] = {}
        self.loaded_count = 0

    def load_blank_rois(self, template_id: str) -> dict[str, np.ndarray]:
        """Load all blank ROIs for a template."""
        if template_id in self.cache:
            return self.cache[template_id]

        blank_rois_dir = self.templates_dir / template_id / "blank_rois"

        if not blank_rois_dir.exists():
            logger.warning(f"Blank ROIs directory not found: {blank_rois_dir}")
            self.cache[template_id] = {}
            return {}

        blank_rois = {}
        for png_file in blank_rois_dir.glob("*.png"):
            field_id = png_file.stem
            roi_image = cv2.imread(str(png_file))

            if roi_image is None:
                logger.error(f"Failed to load blank ROI: {png_file}")
                continue

            blank_rois[field_id] = roi_image

        self.cache[template_id] = blank_rois
        self.loaded_count += len(blank_rois)

        logger.info(f"Loaded {len(blank_rois)} blank ROIs for template '{template_id}'")

        return blank_rois

    def get_blank_roi(self, template_id: str, field_id: str) -> np.ndarray | None:
        """Get specific blank ROI."""
        if template_id not in self.cache:
            self.load_blank_rois(template_id)

        return self.cache.get(template_id, {}).get(field_id)

    def get_loaded_count(self) -> int:
        """Return total number of loaded blank ROIs."""
        return self.loaded_count
```

2. Integrate into `vlm_pdf_recognizer/pipeline.py` (DocumentProcessor):

```python
class DocumentProcessor:
    def __init__(self, templates_dir: str, verbose: bool = False):
        # ... existing initialization ...

        # NEW: Load blank template ROI cache
        self.blank_roi_cache = BlankTemplateROICache(templates_dir)

        # Pre-load all blank ROIs
        for template_id in self.template_ids:
            self.blank_roi_cache.load_blank_rois(template_id)
```

**Test**: Verify cache loading:
```python
from vlm_pdf_recognizer.alignment.blank_template_roi_cache import BlankTemplateROICache

cache = BlankTemplateROICache("data")
cache.load_blank_rois("contractor_1")

signature_roi = cache.get_blank_roi("contractor_1", "signature")
assert signature_roi is not None
assert signature_roi.shape[0] > 0  # Has height
assert signature_roi.shape[1] > 0  # Has width
print(f"Loaded {cache.get_loaded_count()} blank ROIs")
```

---

### Phase 5: VLM Recognizer Integration (2 hours)

**Goal**: Integrate preprocessing into VLM recognition workflow

**Steps**:

1. Modify `vlm_pdf_recognizer/recognition/vlm_recognizer.py`:

```python
class VLMRecognizer:
    def _recognize_field(
        self,
        roi_image: np.ndarray,
        field_schema: FieldSchema,
        template_id: str = None,
        blank_roi_cache = None,  # BlankTemplateROICache
        max_retries: int = 3
    ) -> RecognitionResult:
        """Recognize content with preprocessing integration."""

        # Skip preprocessing for title and checkbox fields
        if field_schema.field_type in ["title", "checkbox"]:
            # ... existing logic (unchanged) ...
            return RecognitionResult(
                field_id=field_schema.field_id,
                # ... existing fields ...
                preprocessing_has_content=None,  # Skipped
                preprocessing_ink_ratio=None,
                preprocessing_component_count=None,
                preprocessing_time_ms=None
            )

        # Step 1: Preprocessing pipeline (NEW)
        preprocessing_result = None
        if blank_roi_cache and template_id:
            blank_roi = blank_roi_cache.get_blank_roi(template_id, field_schema.field_id)

            if blank_roi is not None:
                from vlm_pdf_recognizer.recognition.roi_preprocessor import ROIPreprocessor

                preprocessor = ROIPreprocessor(
                    save_debug_images=config.DEBUG_SAVE_INTERMEDIATE_IMAGES,
                    output_dir="output/processed_rois"
                )

                preprocessing_result = preprocessor.preprocess_roi(
                    roi_image, blank_roi, field_schema.field_id
                )

        # Step 2: VLM inference (always run for content extraction)
        # ... existing VLM logic (unchanged) ...

        # Step 3: Return combined result
        return RecognitionResult(
            field_id=field_schema.field_id,
            # ... existing VLM fields ...
            preprocessing_has_content=preprocessing_result.has_content if preprocessing_result else None,
            preprocessing_ink_ratio=preprocessing_result.ink_ratio if preprocessing_result else None,
            preprocessing_component_count=preprocessing_result.component_count if preprocessing_result else None,
            preprocessing_time_ms=preprocessing_result.processing_time_ms if preprocessing_result else None
        )
```

2. Add preprocessing fields to RecognitionResult dataclass:

```python
@dataclass
class RecognitionResult:
    # ... existing fields ...

    # NEW: Preprocessing fields
    preprocessing_has_content: bool | None = None
    preprocessing_ink_ratio: float | None = None
    preprocessing_component_count: int | None = None
    preprocessing_time_ms: float | None = None
```

3. Update DocumentRecognitionOutput.calculate_results_status():

```python
def calculate_results_status(self) -> bool:
    """Use preprocessing_has_content as primary validation input."""

    # VX1 priority (use heuristic, not preprocessing)
    vx1_result = next((r for r in self.field_results if r.field_id == "VX1"), None)
    if vx1_result and vx1_result.has_content:
        return False

    # Date fields OR (use preprocessing, fallback to VLM)
    date_fields = [r for r in self.field_results if r.field_id in ("year", "month", "date")]
    if date_fields:
        date_valid = any(
            r.preprocessing_has_content if r.preprocessing_has_content is not None else r.has_content
            for r in date_fields
        )
    else:
        date_valid = True

    # Other fields AND (use preprocessing, fallback to VLM)
    excluded_ids = {"VX1", "year", "month", "date"}
    other_fields = [
        r for r in self.field_results
        if r.has_content is not None and r.field_id not in excluded_ids
    ]
    other_fields_valid = all(
        r.preprocessing_has_content if r.preprocessing_has_content is not None else r.has_content
        for r in other_fields
    )

    return date_valid and other_fields_valid
```

**Test**: Integration test with real document:

```bash
# Run main.py with a test document
python main.py

# Verify output includes preprocessing columns
cat output/VLM_results.json | grep "preprocessing_has_content"
```

---

### Phase 6: Output Export (1 hour)

**Goal**: Export preprocessing results in JSON/CSV output

**Steps**:

1. Modify `vlm_pdf_recognizer/output.py` - Update `to_json_dict()`:

```python
def to_json_dict(self) -> Dict[str, Any]:
    """Convert to dictionary with preprocessing results."""
    fields = {}
    for result in self.field_results:
        if result.has_content is None:  # Title field
            continue

        field_data = {
            "VLM_has_content": result.has_content,
            "content_text": result.content_text,
            "AUX_has_content": result.auxiliary_has_content,
            # NEW: Preprocessing results
            "preprocessing_has_content": result.preprocessing_has_content,
            "preprocessing_ink_ratio": result.preprocessing_ink_ratio,
            "preprocessing_component_count": result.preprocessing_component_count
        }

        fields[result.field_id] = field_data

    return {
        "document_ID": str(document_id),
        "results": bool(self.results),
        "type": str(self.template_id),
        "title": str(title_value),
        "processing_timestamp": self.processing_timestamp.isoformat(),
        "fields": fields
    }
```

2. Update CSV export to include preprocessing columns

**Test**: Verify output format:

```bash
python main.py

# Check JSON structure
python -m json.tool output/VLM_results.json | less

# Check CSV columns
head -1 output/vlm_recognition_results.csv
# Should include: preprocessing_*_has_content columns
```

---

### Phase 7: Visualization (1 hour)

**Goal**: Color-code ROI boxes based on preprocessing results

**Steps**:

1. Modify `vlm_pdf_recognizer/output.py` - Update `save_vlm_visualization()`:

```python
def save_vlm_visualization(result, vlm_output, output_dir):
    """Draw ROI boxes colored by preprocessing has_content."""
    # ... existing code ...

    for field_result, roi in zip(vlm_output.field_results, result.extracted_rois):
        # Determine color based on preprocessing result (or fallback)
        if field_result.field_id in ["title"]:
            color = (0, 255, 0)  # Green for title (not validated)
        elif field_result.preprocessing_has_content is not None:
            # Use preprocessing result
            color = (0, 255, 0) if field_result.preprocessing_has_content else (0, 0, 255)
        else:
            # Fallback to VLM (checkboxes, preprocessing failed)
            color = (0, 255, 0) if field_result.has_content else (0, 0, 255)

        # Draw box and label
        cv2.rectangle(visualization_image, (x1, y1), (x2, y2), color, 2)
        label = f"{field_result.field_id}: {field_result.preprocessing_has_content or field_result.has_content}"
        cv2.putText(visualization_image, label, (x1, y1 - 5), font, 0.5, color, 1)
```

**Test**: Visual verification:

```bash
python main.py

# Open visualization image
eog output/sample_document_visualization.png
# Green boxes = content detected
# Red boxes = no content detected
```

---

## Testing Strategy

### Unit Tests (tests/unit/)
- `test_roi_preprocessor.py` - Each pipeline step
- `test_blank_roi_cache.py` - Cache loading and retrieval
- `test_config.py` - Configuration validation

### Integration Tests (tests/integration/)
- `test_preprocessing_pipeline.py` - End-to-end with real documents
- `test_vlm_integration.py` - Preprocessing + VLM workflow
- `test_zero_regression.py` - Verify existing features unchanged

### Run All Tests:
```bash
pytest tests/ -v --cov=vlm_pdf_recognizer
```

---

## Debugging Tips

### Enable Debug Image Saving

```bash
# Set environment variable
export DEBUG_ROI_PREPROCESSING=true

# Run processing
python main.py

# Inspect intermediate images
ls -la output/processed_rois/sample_document/company1/
# 01_saturation.png, 02_difference.png, etc.
```

### Tune Thresholds

1. View debug images to identify failure mode
2. Edit `vlm_pdf_recognizer/recognition/config.py`
3. Re-run processing (no code rebuild needed)
4. Iterate until accuracy improves

### Common Issues

**Issue**: Preprocessing detects empty field as filled (false positive)
- **Debug**: Check `02_difference.png` - high noise/artifacts?
- **Fix**: Increase `DIFFERENCE_THRESHOLD` (e.g., 25 → 30)

**Issue**: Preprocessing misses faint signature (false negative)
- **Debug**: Check `01_saturation.png` - signature visible?
- **Fix**: Increase `SATURATION_BOOST_FACTOR` (e.g., 2.0 → 2.5)
- **Fix**: Decrease `INK_RATIO_THRESHOLD` (e.g., 0.005 → 0.003)

**Issue**: Form lines not removed
- **Debug**: Check `03_lines_removed.png` - residual lines?
- **Fix**: Increase `MORPHOLOGY_LINE_RATIO` (e.g., 0.3 → 0.4)

---

## Performance Profiling

### Measure Preprocessing Time

```python
# Add logging in roi_preprocessor.py
logger.info(f"Field {field_id} preprocessing: {processing_time_ms:.1f}ms")
```

### Identify Bottlenecks

```bash
# Run with profiling
python -m cProfile -o profile.stats main.py

# Analyze results
python -m pstats profile.stats
>>> sort cumtime
>>> stats 20
```

### Optimize if Needed

- Pre-allocate buffers (already done)
- Cache morphological kernels (already done)
- Disable debug image saving in production

---

## Rollout Checklist

Before merging to main:

- [ ] All unit tests pass
- [ ] Integration tests pass with real documents
- [ ] Zero regression: Existing VLM/checkbox tests unchanged
- [ ] Performance <100ms per ROI verified
- [ ] Debug images validate pipeline steps
- [ ] Threshold tuning complete for test documents
- [ ] Documentation updated (README, docstrings)
- [ ] Code reviewed

---

## Next Steps

After completing implementation:

1. Run `/speckit.tasks` to generate task breakdown
2. Execute tasks incrementally with testing
3. Iterate on threshold tuning with real documents
4. Consider adding `/speckit.analyze` for quality verification

For questions or issues, refer to:
- [spec.md](spec.md) - Requirements reference
- [research.md](research.md) - Technical decisions
- [data-model.md](data-model.md) - Data structure reference
