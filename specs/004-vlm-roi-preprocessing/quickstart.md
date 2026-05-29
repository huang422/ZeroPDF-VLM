# Quickstart Guide: AIP — ROI Content Detection

**Feature**: 004-vlm-roi-preprocessing
**Last Aligned With Code**: 2026-05-26
**Status**: Reflects current production AIP pipeline (ECC + BGR mean-difference).

This guide walks you through inspecting / debugging the AIP stage of the pipeline. For the broader pipeline, see [`specs/002-vlm-roi-recognition/quickstart.md`](../002-vlm-roi-recognition/quickstart.md).

---

## Prerequisites

- The project already runs end-to-end (`python main.py` works).
- `update_configs.py` has populated `data/<template_id>/blank_rois/<field_id>.png` for every template.

---

## Inspect a Single Document with Debug Images On

```bash
DEBUG_ROI_PREPROCESSING=true python main.py
```

This single env var flips `vlm_recognizer.DEBUG_SAVE_INTERMEDIATE_IMAGES` to `True`. When set, every non-title ROI saves four files under `output/processed_rois/<document_name>/<field_id>/`:

| File | Meaning |
|---|---|
| `00_aligned_doc.png` | The doc ROI **after** ECC sub-pixel alignment. Compare with the original `rois/<base>_roi_<field>.png` (saved separately by Feature 002). |
| `01_diff_gray.png` | The grayscale BGR-mean-diff between aligned doc and blank template. Bright pixels = content. |
| `02_final.png` | Alias of `01_diff_gray.png` (kept for tooling compatibility). |
| `metadata.json` | `has_content`, `ink_ratio`, `processing_time_ms`, `thresholds_used`, `decision_reasoning`, `error_message`. |

In production runs (no env var), only the **per-ROI processed PNG** (`output/<date>/<case_id>/processed_rois/<base>_roi_<field>_processed.png`) is saved — same content as `01_diff_gray.png`.

---

## Diagnostic Recipe — "Why did this field get has_content=X?"

1. Look up the field's `metadata.json` (need `DEBUG_ROI_PREPROCESSING=true`).
2. Read the `decision_reasoning` string. Two forms:
   - `mean_diff=0.0042, threshold=0.0100, has_content=False` → normal branch, very low diff, field considered empty.
   - `mean_diff=0.1832 (HIGH, pre-printed?), significant_ratio=0.2541, threshold=0.20, has_content=True` → pre-printed-text branch, enough significantly-different pixels to call it filled.
3. Cross-check with the images:
   - `00_aligned_doc.png` vs. blank: do they overlay well? If not, ECC didn't converge — see the next section.
   - `01_diff_gray.png` should be near-black for empty fields; bright where content exists.

### "ECC didn't converge"

Set log level to DEBUG to see `ECC alignment failed: ... using original ROI` lines. Common causes:

| Cause | Symptom |
|---|---|
| Doc ROI is severely warped within its bounding box | ECC hits its 50-iter / 0.001 EPS termination without converging. |
| Doc and template ROIs are very different in content | The cost function can't decrease consistently. |
| Doc ROI has too little gradient (large white area) | ECC has no signal to align. |

In all cases, AIP falls back to **unaligned diff**. Results are still usable for highly-textured fields; less reliable for stamp boxes with pre-printed text.

---

## Tuning Thresholds

Two constants in [`vlm_pdf_recognizer/recognition/roi_preprocessor.py`](../../vlm_pdf_recognizer/recognition/roi_preprocessor.py) plus three inline values control AIP decisions:

| Constant / inline | Default | Effect of raising | Effect of lowering |
|---|---|---|---|
| `MIN_ABSOLUTE_DENSITY_THRESHOLD` (module-level) | `0.01` | More false negatives (misses faint content). | More false positives (calls noise as content). |
| `significant_threshold` (inline, 0..255) | `30` | Fewer pixels count as "significant"; high-diff branch becomes more conservative. | More pixels count; high-diff branch fires sooner. |
| `mean_diff > 0.15` (branch threshold) | `0.15` | More fields use the normal branch (`> 0.01`); fields with pre-printed text may false-positive. | More fields use the `significant_ratio` gate; cleaner stamp-box detection but might miss heavy fills. |
| `significant_ratio > 0.20` (high-diff branch) | `0.20` | High-diff branch becomes stricter. | High-diff branch becomes more permissive. |

After any change, run a representative batch and inspect `result_log.md` failures plus a spot-check of `processed_rois/`.

> **Treat changes here as behaviour changes.** Update [`spec.md`](./spec.md) FR-003 / FR-010 in the same commit.

---

## Adding a New Template Type

1. Create the golden image at `templates/images/<id>.jpg`.
2. Annotate ROIs in LabelMe → `templates/location/<id>.json`.
3. Register the template in `vlm_pdf_recognizer/templates/template_loader.load_all_templates()`.
4. Run `python update_configs.py` — this will populate `data/<id>/blank_rois/<field>.png` for the new template.
5. (If the case-level validation must include the new template) update `REQUIRED_TEMPLATE_TYPES` in `vlm_pdf_recognizer/output.py:202` and update FR-013 in `specs/002-vlm-roi-recognition/spec.md`.
6. Run a smoke batch. Confirm AIP fires for the new template's fields by checking `output/.../processed_rois/`.

---

## Common Pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| All AIP results return `has_content=None` for one template | Blank ROIs missing under `data/<id>/blank_rois/` | Re-run `python update_configs.py`. |
| AIP returns `has_content=True` for clearly empty fields | Blank ROI is stale (template image was updated) | Re-run `python update_configs.py`. |
| AIP returns `False` for clearly filled fields | Faint content (e.g. pencil) below `mean_diff = 0.01` | Lower the threshold or rescan at higher DPI. |
| `ink_ratio` is `None` and `error_message` is populated | Exception inside `preprocess_roi` — check `error_message` | Inspect input shape/dtype; rerun with `DEBUG_ROI_PREPROCESSING=true`. |

---

## Performance Reference

Measured on RTX 4080 Laptop / Ubuntu (AIP is CPU-bound):

| Step | Time |
|---|---|
| Resize (if needed) | < 1 ms |
| ECC alignment | 5–20 ms |
| BGR diff + mean | < 5 ms |
| Threshold decision | < 1 ms |
| **Total per ROI** | **10–30 ms** |

A document with 14 ROIs spends ~ 200–400 ms total in AIP — orders of magnitude less than the 5–10 s spent in VLM for the same document.
