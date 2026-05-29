# Quickstart Guide: Document Template Alignment & ROI Extraction

**Feature**: 001-document-template-alignment
**Last Aligned With Code**: 2026-05-26
**Status**: Reflects current `main.py` + `vlm_pdf_recognizer/` behaviour.

This guide walks you through exercising the **alignment-only** stage (without VLM) so you can confirm the template-matching / warp / ROI-extraction pipeline works in isolation. For the full pipeline including VLM recognition, see the project [README.md](../../README.md).

---

## Prerequisites

- Python 3.9+ (project uses a `vlmcv` conda env by convention)
- macOS / Linux / Windows
- No GPU required for this stage (GPU only matters for the downstream VLM stage)

---

## Installation

```bash
git clone https://github.com/huang422/ZeroPDF-VLM
cd ZeroPDF-VLM

conda create -n vlmcv python=3.9
conda activate vlmcv

pip install -r requirements.txt   # opencv, numpy, PyMuPDF, Pillow, opencc, requests
```

You do not need Ollama or any VLM dependencies to exercise this feature in isolation.

---

## Setup Golden Templates

Templates are configured **once** via the bundled config generator, not by hand:

1. **Place a clean golden image** at `templates/images/<template_id>.jpg`.
2. **Annotate ROIs in LabelMe** and save the JSON at `templates/location/<template_id>.json`.
3. **Add the template ID** to `load_all_templates()` in `vlm_pdf_recognizer/templates/template_loader.py`.
4. **Generate configs**:

   ```bash
   python update_configs.py
   ```

   This writes `data/<template_id>/config.json` (ROI coordinates), `data/<template_id>/blank_rois/*.png` (used by Feature 004 AIP), and updates `vlm_pdf_recognizer/recognition/field_schema.py`.

The three production templates (`contractor_1`, `contractor_2`, `enterprise_1`) are already registered.

---

## Input Layout

Inputs **must** follow the nested layout — this is enforced by `main.py:scan_nested_input()`:

```
input/
└── <date>/                      # e.g. 2026-05-26
    └── <case_id>/               # e.g. case_a101
        ├── doc1.pdf
        └── doc2.pdf
```

Supported file extensions: `.pdf`, `.jpg`, `.jpeg`, `.png`.

---

## Run the Alignment Stage Alone

```bash
python main.py --disable-vlm
```

The `--disable-vlm` flag stops the pipeline after ROI extraction (no Ollama call, no AIP comparison-based has_content). It is the cleanest way to verify this feature.

---

## Output Structure

Outputs mirror the input layout, anchored by date:

```
output/
└── <date>/
    ├── VLM_results.json                       # Batch summary (preprocessing-only when --disable-vlm)
    └── <case_id>/
        ├── <doc>_visualization.png            # Aligned image with ROI boxes
        └── metadata/
            └── <doc>_metadata.json            # matched_template_id, inlier count, ROI list
```

When you re-run **without** `--disable-vlm`, the same `<doc>_visualization.png` gets **overwritten** with a colour-coded version (green = has_content / red = no content / blue = unavailable). That overwrite is owned by Feature 002 (`save_vlm_visualization`).

---

## Verification Checklist

For each document, confirm:

1. **`<doc>_visualization.png`** shows the document warped to template dimensions with bounding boxes drawn around every configured ROI.
2. **`metadata/<doc>_metadata.json`** contains:
   - `matched_template_id` in `{contractor_1, contractor_2, enterprise_1}` (or `"unknown"` for rejected docs).
   - `confidence_score` (RANSAC inlier count) ≥ 50 for matched docs.
   - `success: true` for matched docs, `false` for rejected docs.
   - `rois[*]` listing each ROI with `roi_id`, `description`, and `bounding_box`.

---

## Performance Reference

Measured on RTX 4080 Laptop (CPU stage only — GPU isn't used here):

| Step | Time |
|---|---|
| PDF page → BGR | < 100 ms |
| SIFT extraction | 100–300 ms |
| Template matching (FLANN + RANSAC) | 200–500 ms |
| Geometric alignment (warp) | 50–100 ms |
| ROI extraction | < 100 ms |
| **Total per page (alignment stage)** | **< 1 s** |

The 3–5 s "per document" figure in the README is dominated by the **downstream VLM stage**, not this one.

---

## Troubleshooting

### `Unknown Document` (rejected) error

`UnknownDocumentError` fires when the best template has fewer than 50 RANSAC inliers. The error message includes per-template inlier counts.

| Cause | Action |
|---|---|
| Wrong document type | Confirm document is genuinely one of the three templates. |
| Low-quality scan (blur, low DPI) | Rescan at ≥ 200 DPI; SIFT needs feature richness. |
| Severe rotation / cropping | Manually fix before re-processing. |
| Outdated template cache | Delete `data/<template_id>/template_features.pkl` and re-run — features will be recomputed. |

### Wrong template selected

Inspect the per-template inlier breakdown in stdout / error message. If the wrong template wins, the templates are likely too visually similar in the document's overlap region. Consider improving the unique markers on the golden image.

### ROI boxes mis-aligned

The warp is correct only if the homography is correct. If boxes look offset:

| Cause | Action |
|---|---|
| `templates/location/<id>.json` annotated against a different golden image | Re-annotate against the current `templates/images/<id>.jpg`. |
| LabelMe coordinates not regenerated to `config.json` | Re-run `python update_configs.py`. |
| Document genuinely warped (e.g. crinkled paper) | Homography can only correct planar perspective — not crumples. |

### "Failed to load templates" at startup

Means `data/<template_id>/config.json` is missing or unreadable. Re-run `python update_configs.py`.

---

## Next Steps

Once alignment is verified:

1. Re-run **without** `--disable-vlm` to exercise Feature 002 (VLM recognition) and Feature 004 (AIP has_content detection).
2. Inspect `output/<date>/VLM_results.json` and `result_log.md` for case-level validation outcomes.
3. To add a new template, follow the README "Adding New Templates" section.
