# Data Model: AIP — ROI Content Detection

**Feature**: 004-vlm-roi-preprocessing
**Date**: 2025-12-31
**Last Aligned With Code**: 2026-05-26
**Status**: Reflects current production dataclasses in `vlm_pdf_recognizer/recognition/roi_preprocessor.py` and `vlm_pdf_recognizer/alignment/blank_template_roi_cache.py`.

This is the field-level reference. For behaviour, see [spec.md](./spec.md).

> **Drift from the original draft**: the draft defined `PreprocessingConfig` with five threshold knobs and `PreprocessingResult` with metrics for connected components. Production simplified to one `AIPResult` dataclass and two module-level constants. See the `spec.md` "Drift from the original draft" callout.

---

## 1. AIPResult
**Source**: [vlm_pdf_recognizer/recognition/roi_preprocessor.py](../../vlm_pdf_recognizer/recognition/roi_preprocessor.py)

```python
@dataclass
class AIPResult:
    field_id: str
    has_content: Optional[bool]    # True / False / None (None = AIP failed or unavailable)
    ink_ratio: Optional[float]      # = mean_diff in [0.0, 1.0]; None on error
    component_count: Optional[int]  # ALWAYS None in current code (legacy field)
    processing_time_ms: float       # wall-clock, always populated
    error_message: Optional[str] = None
    processed_image: Optional[np.ndarray] = None   # grayscale diff image, for output PNG
```

| Field | Constraint |
|---|---|
| `has_content == False` | implies `ink_ratio` is a valid float |
| `has_content is None` | implies AIP failed or blank ROI missing — `ink_ratio` may be None and `error_message` may be populated |
| `component_count` | Reserved; not used by any decision logic. Always `None`. |
| `processed_image` | The grayscale diff image (after BGR mean diff). Single channel, uint8, same shape as the template ROI. |

---

## 2. ROIPreprocessor
**Source**: same file.

```python
class ROIPreprocessor:
    def __init__(
        self,
        save_debug_images: bool = False,
        output_dir: Optional[Path] = None,
    ): ...

    def preprocess_roi(
        self,
        doc_roi_image: np.ndarray,     # BGR uint8 (H, W, 3)
        blank_template_roi: np.ndarray, # BGR uint8 (H, W, 3)
        field_id: str,
        document_name: Optional[str] = None,
    ) -> AIPResult: ...

    # Internal helpers:
    def _align_roi_to_template(...) -> np.ndarray
    def _save_intermediate_images(...) -> None      # only if save_debug_images=True
    def _save_metadata(...) -> None                  # only if save_debug_images=True
```

**Thread safety**: NOT thread-safe — reuses internal buffers. Use one instance per thread.

**Construction**: `Feature 002` constructs a fresh `ROIPreprocessor` per ROI within `_recognize_field` (see `vlm_recognizer.py:_recognize_field` → `from .roi_preprocessor import ROIPreprocessor; preprocessor = ROIPreprocessor(...)`). This is intentional: AIP setup is cheap.

---

## 3. Module-Level Constants
**Source**: top of `roi_preprocessor.py`.

```python
MIN_ABSOLUTE_DENSITY_THRESHOLD = 0.01     # mean_diff threshold for "normal" fields
PNG_COMPRESSION_LEVEL          = 6        # for debug image writes
```

Two more constants appear inline inside `preprocess_roi` and are part of the contract (changing them is a behaviour change):

| Inline | Value | Purpose |
|---|---|---|
| `significant_threshold` | `30` | Per-pixel diff threshold (0..255) used to compute `significant_ratio`. |
| `mean_diff > 0.15` | `0.15` | Branch threshold — above this, the field likely has pre-printed text, so use `significant_ratio` decision instead of `mean_diff` decision. |
| `significant_ratio > 0.20` | `0.20` | High-mean-diff branch: ≥ 20% significantly-different pixels → has content. |

---

## 4. BlankTemplateROICache
**Source**: [vlm_pdf_recognizer/alignment/blank_template_roi_cache.py](../../vlm_pdf_recognizer/alignment/blank_template_roi_cache.py)

```python
class BlankTemplateROICache:
    def __init__(self): ...
    def load_blank_rois(self, template_id: str, templates_dir: str) -> int: ...
    def get_blank_roi(self, template_id: str, field_id: str) -> Optional[np.ndarray]: ...
    def get_loaded_templates(self) -> List[str]: ...
```

**Storage layout**: A nested dict keyed by `(template_id, field_id)` mapping to a BGR uint8 ndarray loaded from `data/<template_id>/blank_rois/<field_id>.png`.

**Population**: Called once per template by `DocumentProcessor.__init__` → `blank_template_roi_cache.load_blank_rois(template_id, templates_dir)`. Returns the count of blank ROI files successfully loaded.

**Failure mode**: If a blank PNG is missing, the cache logs and continues; `get_blank_roi()` returns `None` for that key, and AIP is skipped for that field at runtime.

---

## 5. Debug Output (when `DEBUG_ROI_PREPROCESSING=true`)

Structure under `output/processed_rois/<document_name>/<field_id>/`:

```
00_aligned_doc.png    # doc ROI after ECC alignment (BGR)
01_diff_gray.png      # grayscale BGR-mean-diff
02_final.png          # alias of 01_diff_gray.png (kept for visual review tools)
metadata.json         # see schema below
```

`metadata.json` shape:
```jsonc
{
  "field_id": "person1",
  "has_content": true,
  "ink_ratio": 0.0421,
  "component_count": null,
  "processing_time_ms": 18.3,
  "thresholds_used": {
    "MIN_ABSOLUTE_DENSITY_THRESHOLD": 0.01
  },
  "decision_reasoning": "mean_diff=0.0421, threshold=0.0100, has_content=True",
  "error_message": null
}
```

The decision_reasoning string also captures the high-mean-diff branch:
```
"mean_diff=0.1832 (HIGH, pre-printed?), significant_ratio=0.2541, threshold=0.20, has_content=True"
```

## 6. Production Output (always)

Feature 002 saves a single PNG per ROI regardless of the debug flag, via `output.save_preprocessed_rois`:

```
output/<date>/<case_id>/processed_rois/<base>_roi_<field>_processed.png
```

This image is `AIPResult.processed_image` (the grayscale diff).

---

## What's Intentionally Absent

- **No `PreprocessingConfig` dataclass** — production has two module-level constants and three inline numbers. Refactor to a config class only if multiple template-specific configs become necessary.
- **No `MIN_BLOB_AREA` / `COMPONENT_COUNT_THRESHOLD` / `MORPHOLOGY_LINE_RATIO`** — those knobs were proposed in the draft but never adopted.
- **No HSV preprocessing**. The pipeline operates directly in BGR.
- **No checkbox-specific heuristic**. Checkboxes go through the same AIP pipeline as text / stamps / numbers.
