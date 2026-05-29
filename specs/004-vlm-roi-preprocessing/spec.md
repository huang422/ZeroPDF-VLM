# Feature Specification: AIP (Advanced Image Processing) — ROI Content Detection

**Feature Branch**: `004-vlm-roi-preprocessing`
**Created**: 2025-12-31
**Last Aligned With Code**: 2026-05-26
**Status**: Implemented — this spec reflects current production code in `vlm_pdf_recognizer/recognition/roi_preprocessor.py` and `vlm_pdf_recognizer/alignment/blank_template_roi_cache.py`.

---

## Scope Recap (Current Implementation)

This feature is the **pixel-level `has_content` detector**. For each non-title field, AIP compares the document ROI against the **blank reference ROI** for that template / field, and returns a deterministic boolean. Feature 002 (VLM) then handles text extraction.

```
ExtractedROI (Feature 001)            Blank ROI image
       │                                    │ (from BlankTemplateROICache, populated by update_configs.py)
       ▼                                    │
  ┌────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ ROIPreprocessor.preprocess_roi                       │
│                                                      │
│  1. Validate inputs (uint8, 3-channel BGR)           │
│  2. Resize doc ROI to template ROI shape if needed   │
│     (log warning if size diff > 10%)                 │
│  3. ECC sub-pixel alignment (MOTION_EUCLIDEAN)       │
│     → aligned doc ROI                                 │
│  4. BGR absolute difference                          │
│  5. Mean across channels → grayscale diff image      │
│  6. mean_diff = mean(diff_gray) / 255.0              │
│  7. Decision logic (next section)                    │
│                                                      │
│  → AIPResult { has_content, ink_ratio (mean_diff),   │
│                processing_time_ms, processed_image } │
└─────────────────────────────────────────────────────┘
```

### Decision Logic (Current Code)

```python
significant_threshold = 30                       # per-pixel diff threshold (0..255)
significant_diff = diff_gray[diff_gray > significant_threshold]
significant_ratio = len(significant_diff) / diff_gray.size

if mean_diff > 0.15:
    # Heuristic: high mean diff often means pre-printed text dominated
    # the ROI (e.g. big1 with "負責人蓋章處" pre-printed text).
    # Require ≥ 20% pixels above per-pixel threshold to count as actual content.
    has_content = significant_ratio > 0.20
else:
    # Normal field path
    has_content = mean_diff > MIN_ABSOLUTE_DENSITY_THRESHOLD  # 0.01
```

> **Drift from the original draft of this spec** (kept here for traceability):
> - The draft specified a **6-step pipeline** (HSV saturation → grayscale diff → horizontal/vertical line morphology → noise removal → connected components → ink-ratio threshold). Production is **simpler**: ECC align → BGR mean diff → 2-tier threshold. The HSV channel, morphology, and connected-components stages were removed because (a) BGR-mean-diff was already discriminating well, (b) morphology added per-template tuning burden, (c) connected-components count added latency without clear accuracy gains.
> - The draft specified 5 configurable thresholds (`DIFFERENCE_THRESHOLD`, `INK_RATIO_THRESHOLD`, `COMPONENT_COUNT_THRESHOLD`, `MIN_BLOB_AREA`, `MORPHOLOGY_LINE_RATIO`). Production has **two** constants: `MIN_ABSOLUTE_DENSITY_THRESHOLD = 0.01` and `significant_threshold = 30` (plus the inline `mean_diff > 0.15` and `significant_ratio > 0.20`).
> - The draft specified saving 6 intermediate PNGs per ROI. Production saves **three**: `00_aligned_doc`, `01_diff_gray`, `02_final` (only when `DEBUG_ROI_PREPROCESSING=true` env var is set).
> - The draft used `component_count` for the threshold. Production keeps the `component_count` slot in `AIPResult` for backwards compatibility but **leaves it `None`** — see `roi_preprocessor.py:219`.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Detect Filled vs Empty Fields (Priority: P1)

A user runs the pipeline on a populated `contractor_1` PDF. AIP needs to mark filled fields (name, company, stamp) as `has_content=True` and untouched fields (e.g. an unmarked checkbox, a date field left blank) as `False`. The VLM then runs only on the True ones.

**Independent Test**: Process a PDF where `person1` is filled with handwriting and `month` is blank. In the resulting `RecognitionResult` for those fields, confirm:
- `AIP_has_content` matches expectation.
- `AIP_ink_ratio` (mean diff, 0.0–1.0) is much higher for `person1` than `month`.
- For the empty `month`, VLM is **skipped** (`content_text=None`, `inference_time_ms=0`).

**Acceptance Scenarios**:

1. **Given** a blank stamp box (matches template), **When** AIP runs, **Then** `mean_diff < 0.01` → `has_content=False`.
2. **Given** a stamp box with a red company seal, **When** AIP runs, **Then** `mean_diff > 0.01` → `has_content=True`.
3. **Given** a stamp box with **only pre-printed placeholder text** `負責人蓋章處` (no actual stamp), **When** the document ROI happens to come out slightly mis-aligned (residual local registration error after Feature 001's homography), **Then** ECC re-alignment minimises text overlap error and the high-mean-diff branch (`mean_diff > 0.15`) checks `significant_ratio` — `False` if the residual is just text outline noise, `True` if the actual stamp ink dominates.

---

### User Story 2 — Be Robust to Sub-Pixel Misalignment (Priority: P1)

Feature 001's homography is global per page. Within a single ROI there can still be sub-pixel translation / rotation drift, especially when the scanner introduced local warping. AIP must correct for that before differencing — otherwise pre-printed text will "ghost" into the diff and create false positives.

**Independent Test**: Compare `00_aligned_doc.png` and the original cropped ROI (in `rois/`). They should differ by a tiny translation; the aligned version should overlay the blank template ROI better.

**Acceptance Scenarios**:

1. **Given** a document ROI translated by 2–3 px relative to the template (typical scanner drift), **When** ECC alignment runs, **Then** the `aligned_doc` is shifted to overlay the template, and `mean_diff` for a genuinely blank field drops below 0.01.
2. **Given** an ROI where ECC fails to converge (`cv2.error`), **When** the exception is caught, **Then** the original ROI is used and processing continues with a debug log line `ECC alignment failed: ... using original ROI`.
3. **Given** doc ROI shape ≠ template ROI shape, **When** the size difference is > 10%, **Then** a warning is logged; the doc ROI is resized to template shape with `INTER_LINEAR` before continuing.

---

### User Story 3 — Skip VLM on Detected-Empty Fields (Priority: P1)

The whole point of AIP is to avoid wasting 0.5–1.5 s of GPU time on fields that are clearly blank. When AIP says `has_content=False`, Feature 002 must skip the VLM call.

**Independent Test**: Process a document with 5 empty fields. Check the VLM inference time total: it should be ~5 × (VLM-per-roi) less than a document with all fields filled.

**Acceptance Scenarios**:

1. **Given** `aip_result.has_content=False` for a non-checkbox field, **When** Feature 002 routes the field, **Then** VLM is **skipped** and `RecognitionResult.content_text=None`, `inference_time_ms=0`.
2. **Given** `aip_result.has_content=True`, **When** Feature 002 routes the field, **Then** VLM runs and `content_text` is populated (subject to its own cleanup rules).
3. **Given** AIP raises an exception (e.g. corrupt blank ROI), **When** the failure is caught, **Then** `RecognitionResult.has_content=None` (AIP unavailable) and VLM still runs because the falsy-skip predicate is `has_content is False`, not "not True".

---

### User Story 4 — Configurable Blank-ROI Generation (Priority: P1)

`update_configs.py` produces the blank reference ROIs from `templates/images/<id>.jpg` using LabelMe ROI coordinates. Re-running it after annotation changes must regenerate every blank ROI under `data/<id>/blank_rois/`.

**Independent Test**: Run `python update_configs.py`, then run the pipeline. Inspect `data/<id>/blank_rois/<field>.png` and confirm it shows the **pre-printed** form region for that field (no user content, since the source is the clean golden template).

**Acceptance Scenarios**:

1. **Given** a fresh template image + LabelMe annotation, **When** `update_configs.py` runs, **Then** `data/<id>/blank_rois/<field>.png` exists for every non-title ROI.
2. **Given** updated LabelMe coordinates, **When** `update_configs.py` re-runs, **Then** existing blank PNGs are overwritten.
3. **Given** a blank ROI file is missing at pipeline start time, **When** `BlankTemplateROICache.load_blank_rois()` runs, **Then** the missing entry is logged and the cache still loads the others.

---

### User Story 5 — Inspectable Diff Images (Priority: P2)

For debugging false positives/negatives, the `processed_image` (the grayscale diff after ECC alignment) is bundled into the `RecognitionResult` and saved by Feature 002 as `processed_rois/<base>_roi_<field>_processed.png`.

**Independent Test**: After a run, open any processed ROI PNG. A genuinely filled field should show bright pixels where user content existed; a genuinely empty field should be near-black overall.

**Acceptance Scenarios**:

1. **Given** any non-title field after a successful run, **When** the user opens `processed_rois/<base>_roi_<field>_processed.png`, **Then** they see the grayscale BGR-mean-diff image.
2. **Given** `DEBUG_ROI_PREPROCESSING=true` is set in the environment, **When** the pipeline runs, **Then** additional intermediates (`00_aligned_doc.png`, `01_diff_gray.png`, `02_final.png`) and a `metadata.json` (with `decision_reasoning`) are saved under `output/processed_rois/<document_name>/<field_id>/`.
3. **Given** `DEBUG_ROI_PREPROCESSING` is unset (default), **When** the pipeline runs, **Then** only the per-field final PNG is saved (no intermediates).

---

### Edge Cases (Behaviour Today)

| Scenario | Current Behaviour |
|---|---|
| Blank ROI image missing for the matched template | AIP is skipped for that field; `aip_result is None`; `has_content` is `None`; VLM still runs. |
| Doc ROI is heavily skewed within the bounding box | ECC `MOTION_EUCLIDEAN` (translation + rotation, 4 DoF) corrects small misalignment; severe skew makes ECC fail → falls back to unaligned diff (still usable in most cases). |
| Doc ROI has very different content than template (e.g. wrong document type slipped through Feature 001) | `mean_diff` will be very high; the `mean_diff > 0.15` branch fires; `significant_ratio` is computed but unreliable since the entire ROI is "different". Feature 001 should have caught this earlier with low inliers. |
| User content with low contrast (faint pencil) | Mean diff may be just above 0.01 → `has_content=True` (correct). Very faint content (mean diff < 0.005) is missed (acceptable for production use case). |
| BGR colour channels saturated by JPEG compression | No special handling; BGR mean diff smooths over channel-specific noise. |

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `update_configs.py` MUST extract blank reference ROI images from `templates/images/<template_id>.jpg` using the bounding boxes in `templates/location/<template_id>.json` (LabelMe), saving each to `data/<template_id>/blank_rois/<field_id>.png`. Title fields and version fields do **not** require blank ROIs (they bypass AIP).
- **FR-002**: `BlankTemplateROICache.load_blank_rois(template_id, templates_dir)` MUST load every PNG in `data/<template_id>/blank_rois/` into memory at pipeline startup. The cache MUST report the count of loaded blanks. Missing files are logged but do not abort startup.
- **FR-003**: `ROIPreprocessor.preprocess_roi(doc_roi, blank_template_roi, field_id, document_name)` MUST execute the following pipeline in order:
  1. Validate that both inputs are `uint8` 3-channel BGR ndarrays. Raise `ValueError` otherwise.
  2. If shapes differ, log a warning when the relative difference exceeds 10%, then resize doc ROI to template shape using `cv2.resize` with `INTER_LINEAR` (default).
  3. ECC align doc ROI to template ROI using grayscale conversion, `cv2.MOTION_EUCLIDEAN`, termination criteria `(TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 50, 0.001)`, `gaussFiltSize=1`. On `cv2.error`, log and use the original ROI.
  4. Cast both ROIs to `float32` and compute `abs_diff = np.abs(doc_float - template_float)`.
  5. Reduce to grayscale with `diff_gray = mean(abs_diff, axis=2).astype(uint8)`.
  6. Compute `mean_diff = mean(diff_gray) / 255.0`.
  7. Compute `significant_diff = diff_gray[diff_gray > 30]` and `significant_ratio = len(significant_diff) / diff_gray.size`.
  8. Decision: `has_content = significant_ratio > 0.20` if `mean_diff > 0.15`, else `has_content = mean_diff > 0.01`.
  9. Return an `AIPResult` with `has_content`, `ink_ratio=mean_diff`, `component_count=None`, `processing_time_ms`, `processed_image=diff_gray`.
- **FR-004**: AIP MUST be invoked **only** for fields where `field_type` is not `"title"` and not `"version"`. Title fields skip AIP entirely (`has_content` stays `None`); version fields skip AIP and have `has_content` hard-coded to `True` by Feature 002.
- **FR-005**: When the blank ROI is unavailable for a given `(template_id, field_id)` pair, AIP MUST be skipped silently for that field; `RecognitionResult.has_content = None`. Feature 002 then runs VLM unconditionally for that field.
- **FR-006**: AIP MUST handle its own exceptions and return an `AIPResult` with `has_content=None`, `ink_ratio=None`, `component_count=None`, `error_message=str(e)` rather than propagating. The pipeline continues with the next field.
- **FR-007**: Per-field debug images MUST be saved **only when** the environment variable `DEBUG_ROI_PREPROCESSING=true` is set (see `vlm_recognizer.py:DEBUG_SAVE_INTERMEDIATE_IMAGES`). When saved, they go to `output/processed_rois/<document_name>/<field_id>/`:
  - `00_aligned_doc.png` — doc ROI after ECC alignment.
  - `01_diff_gray.png` — grayscale BGR mean-diff image.
  - `02_final.png` — same as `01_diff_gray.png` (kept for compatibility / future expansion).
  - `metadata.json` — includes `has_content`, `ink_ratio`, `component_count`, `processing_time_ms`, `thresholds_used.MIN_ABSOLUTE_DENSITY_THRESHOLD`, `decision_reasoning`, `error_message`.
- **FR-008**: The **default (production) save behaviour** is handled by Feature 002 via `save_preprocessed_rois`: one PNG per ROI at `output/<date>/<case_id>/processed_rois/<base>_roi_<field>_processed.png`. This is independent of the debug flag.
- **FR-009**: `ROIPreprocessor` MUST NOT cache state between calls; the same instance can be reused across documents but is **not thread-safe** (the underlying ECC buffers are reused). For parallel processing, create one instance per thread.
- **FR-010**: Production thresholds (`MIN_ABSOLUTE_DENSITY_THRESHOLD = 0.01`, `significant_threshold = 30`, `mean_diff > 0.15` branch, `significant_ratio > 0.20`) MUST be module-level constants in `roi_preprocessor.py`. Any change to these constants is a behaviour change and MUST update this spec.

### Key Entities

See [data-model.md](./data-model.md) for full schemas.

| Entity | Module | Role |
|---|---|---|
| `BlankTemplateROICache` | `alignment/blank_template_roi_cache.py` | Loads + stores blank reference ROI images in memory; queried via `get_blank_roi(template_id, field_id)`. |
| `ROIPreprocessor` | `recognition/roi_preprocessor.py` | Executes the ECC + BGR-diff pipeline; returns `AIPResult`. |
| `AIPResult` | same | Output dataclass: `has_content`, `ink_ratio` (= `mean_diff`), `processing_time_ms`, `processed_image`, `error_message`. |
| `DEBUG_SAVE_INTERMEDIATE_IMAGES` (env-driven flag) | `vlm_recognizer.py` | When True, AIP saves intermediates + metadata.json under `output/processed_rois/`. |
| `update_configs.py` (top-level script) | repo root | Generates `data/<id>/blank_rois/<field>.png` from LabelMe annotations. |

---

## Success Criteria *(mandatory)*

Measured on RTX 4080 Laptop / Ubuntu:

- **SC-001**: AIP per-ROI latency is **10–30 ms** (dominated by ECC).
- **SC-002**: AIP correctly classifies **unambiguous** filled vs empty fields (clear handwriting / no handwriting; stamped / not stamped) with high accuracy on the production document set. There is no formal benchmark in `tests/`; accuracy is monitored qualitatively via `result_log.md` failures.
- **SC-003**: Skipping VLM on AIP-empty fields measurably reduces per-document time (verifiable by comparing total VLM time before/after disabling AIP — disabling is **not** a supported config; this is purely observational).
- **SC-004**: When `DEBUG_ROI_PREPROCESSING=true`, every non-title field gets a 3-image debug bundle plus `metadata.json` with the decision reasoning string.
- **SC-005**: Pipeline never crashes due to AIP failures — failures are caught, logged, and converted to `has_content=None`.

---

## Assumptions

1. **Blank template golden image is clean** — no annotations, no handwriting. AIP cannot distinguish "pre-existing handwriting on template" from "user content".
2. **Homography from Feature 001 is approximately correct** — ECC corrects sub-pixel drift, not global misalignment. If the homography is fundamentally wrong (e.g. wrong template selected), AIP results are meaningless.
3. **`MIN_ABSOLUTE_DENSITY_THRESHOLD = 0.01` is calibrated for the current document set.** Other document types may need re-tuning.
4. **The `mean_diff > 0.15` branch targets fields with pre-printed placeholder text** like stamp boxes (`負責人蓋章處`). On such fields, residual text after ECC alignment can elevate the mean diff; only a high fraction of significant pixels (≥ 20%) indicates real content.
5. **`update_configs.py` is run after every template / annotation change.** Stale blank ROIs will silently produce wrong `has_content` decisions.

---

## Out of Scope

- HSV / saturation channel preprocessing (was in the draft; not adopted).
- Morphological line removal (horizontal/vertical kernels) — not adopted.
- Connected-component analysis with `MIN_BLOB_AREA` threshold — not adopted (`component_count` field stays `None`).
- Per-field-type threshold tuning — single global thresholds only.
- Adaptive / learned thresholds — manual configuration only.
- AIP on checkbox fields — checkboxes use the AIP pipeline too (they are non-title, non-version), just the same path as other fields; the draft's "checkboxes need a separate heuristic" is no longer true.
- AIP on title fields — title fields bypass the entire recognition stack (`predefined_value` is returned).
- Real-time / streaming AIP.
- GPU acceleration of the AIP pipeline (currently CPU OpenCV; latency is fine).

---

## Dependencies

- **OpenCV ≥ 4.8** — `cvtColor`, `findTransformECC`, `warpAffine`, `imencode`, `resize`.
- **NumPy** — pixel arithmetic.
- **Feature 001** — provides aligned ROIs and the matched template ID.
- **Feature 002** (`vlm_recognizer.py`) — invokes AIP via `_recognize_field`, consumes `AIPResult.has_content`.
- **`update_configs.py`** — produces blank ROI PNGs that AIP relies on.

---

## Notes for Spec-vs-Code Reviewers

- The `AIPResult.component_count` field is **vestigial** — kept in the dataclass for backwards binary compatibility with old JSON outputs, always `None`. Do not add new logic that reads it.
- The `processed_image` returned in `AIPResult` is the grayscale diff (`diff_gray`), not a binarised mask. The output PNG is therefore continuous-tone — bright where content was detected.
- ECC failure is **not** a fatal error — code logs at DEBUG level and continues with the unaligned ROI. If your debug logs are at INFO level you will miss these. Bump to DEBUG when investigating odd `has_content` outcomes.
- The `mean_diff > 0.15` constant was empirically chosen against the production stamp / signature templates. Treat it as a tuning knob, not a universal threshold.
