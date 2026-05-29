# Implementation Plan: AIP — ROI Content Detection

**Branch**: `004-vlm-roi-preprocessing` | **Date**: 2025-12-31 | **Spec**: [spec.md](spec.md)
**Last Aligned With Code**: 2026-05-26
**Status**: Implemented. This plan documents the production AIP pipeline (replacing the original 6-step draft with the simpler ECC + BGR-diff design).

## Summary

Provide a deterministic, pixel-based `has_content` detector for non-title fields, so that Feature 002 (VLM recognition) can skip clearly-empty ROIs and rely on AIP — not the VLM — for the boolean fed into validation logic.

Production pipeline per ROI:

1. Validate inputs (uint8 BGR, 3-channel).
2. Resize doc ROI to template shape if dimensions differ (warn at > 10%).
3. ECC sub-pixel alignment (MOTION_EUCLIDEAN) — corrects residual local drift after Feature 001's global homography.
4. BGR absolute difference between aligned doc ROI and blank template ROI.
5. Reduce to grayscale via mean-across-channels → `diff_gray`.
6. Two-tier threshold:
   - **Normal branch**: `mean_diff > 0.01` ⇒ `has_content=True`.
   - **Pre-printed-text branch**: if `mean_diff > 0.15`, require `significant_ratio > 0.20` (≥ 20% of pixels above per-pixel threshold of 30/255).

`has_content` from AIP drives both per-document validation (`calculate_results_status` in Feature 002) and the visualisation colour code.

## Technical Context

| Item | Value |
|---|---|
| Language | Python 3.9+ |
| Image processing | OpenCV (cvtColor, findTransformECC, warpAffine, resize) |
| Numerics | NumPy |
| Hardware | CPU only — latency 10–30 ms / ROI |
| Failure mode | All AIP exceptions caught and converted to `has_content=None` → VLM still runs |
| Determinism | Fully deterministic given fixed thresholds and blank ROI files |

## Constitution Check

- **Local-first**: ✅ CPU-only, no network calls.
- **Zero training**: ✅ Pure image processing, no model.
- **Determinism**: ✅ Same inputs always yield same `AIPResult`.
- **Graceful degradation**: ✅ Failures yield `has_content=None`, pipeline continues.
- **Explainability**: ✅ `decision_reasoning` string written to debug `metadata.json`.

## Major Pivots From the Original Draft

The first draft proposed a **6-step pipeline** with HSV preprocessing, morphological line removal, connected components, and 5 configurable thresholds. Production simplified to ECC align + BGR mean diff + 2-tier threshold. Reasons:

1. **HSV saturation channel** didn't help enough on the production document set — most fills are dark ink on white background, where BGR-mean-diff already discriminates well.
2. **Morphological line removal** was sensitive to template-specific kernel sizing and could remove faint signatures.
3. **Connected components** added latency (~ 5 ms / ROI) without measurable accuracy gain over the simpler thresholds.
4. **Five configurable thresholds** became a maintenance burden — different templates wanted different values, leading to per-template tuning.

The `mean_diff > 0.15` branch was added as a special case for **fields with pre-printed placeholder text** (e.g. stamp boxes labelled `負責人蓋章處`). On those fields, even a perfectly-aligned blank-template diff shows residual text outline noise; using `significant_ratio` as the secondary gate makes the decision robust to that noise.

The legacy `component_count` field is retained in `AIPResult` for backwards compatibility with debug tooling but is always `None`.

## Project Structure

```text
vlm_pdf_recognizer/
├── alignment/
│   └── blank_template_roi_cache.py   # In-memory cache for blank ROI PNGs
└── recognition/
    └── roi_preprocessor.py            # ROIPreprocessor + AIPResult

update_configs.py                      # Generates data/<id>/blank_rois/<field>.png
data/<template_id>/blank_rois/         # Per-field blank reference PNGs
```

AIP is invoked by `VLMRecognizer._recognize_field` in `vlm_pdf_recognizer/recognition/vlm_recognizer.py` (see `vlm_recognizer.py:_recognize_field` Step 1 — AIP).

## Complexity Notes

- **One `ROIPreprocessor` per `_recognize_field` call.** Construction is cheap; the alternative (singleton) would complicate the debug-image save path.
- **ECC `MOTION_EUCLIDEAN` (4 DoF)** chosen over `MOTION_TRANSLATION` because some scans introduce small rotational drift even after Feature 001's homography.
- **Two-stage threshold is empirically tuned.** Changing either constant is a behaviour change — update the spec and run a representative batch.

## Phase 0 — Research

See [research.md](./research.md). Original research described the 6-step pipeline; the production research is summarised in the "Major Pivots" section above.

## Phase 1 — Design

See [data-model.md](./data-model.md). The original draft's `PreprocessingResult` + `PreprocessingConfig` were merged into a single `AIPResult` with module-level constants.

## Phase 2 — Tasks

See [tasks.md](./tasks.md). All tasks completed; new work follows [DEVELOPMENT_WORKFLOW.md](../../DEVELOPMENT_WORKFLOW.md).

## Maintenance Notes

When changing AIP:

- **Threshold changes** (`MIN_ABSOLUTE_DENSITY_THRESHOLD`, the inline `0.15`, `0.20`, `30`): edit `roi_preprocessor.py` and update FR-003 / FR-010 in `spec.md`.
- **Algorithm changes** (e.g. adding HSV / morphology back): edit `preprocess_roi`, update FR-003 in `spec.md`, and update the pipeline diagram in `spec.md`.
- **Blank ROI generation**: edit `update_configs.py` and ensure `data/<id>/blank_rois/` is regenerated. Existing files become stale on template image changes.

Any change to `AIPResult` schema (new fields, new constraints) MUST propagate to:
- `data-model.md` (this folder).
- `data-model.md` in `specs/002-vlm-roi-recognition/` (since Feature 002 consumes AIP fields).
- `to_json_dict()` in `vlm_recognizer.py` if the new field needs to surface in output JSON.
