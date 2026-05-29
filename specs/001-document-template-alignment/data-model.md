# Data Model: Document Template Alignment & ROI Extraction

**Feature**: 001-document-template-alignment
**Date**: 2025-12-23
**Last Aligned With Code**: 2026-05-26
**Status**: Reflects current dataclasses in `vlm_pdf_recognizer/`. Earlier draft references to "preprocessed_image / watermark removal" have been removed because that preprocessing step was deleted from the pipeline (SIFT handles watermarks natively).

## Overview

This document lists the runtime entities used by the alignment + ROI extraction stage. All entities are Python dataclasses (or plain classes) — no database is involved. Source-of-truth files are linked per entity.

---

## Core Entities

### 1. GoldenTemplate
**Source**: [vlm_pdf_recognizer/templates/__init__.py](../../vlm_pdf_recognizer/templates/__init__.py)

| Field | Type | Notes |
|---|---|---|
| `template_id` | `str` | One of `{contractor_1, contractor_2, enterprise_1}`. |
| `template_image` | `np.ndarray` (BGR, uint8) | Golden image loaded from `templates/images/<id>.jpg`. |
| `image_shape` | `Tuple[int, int, int]` | `(H, W, 3)` — every aligned image is warped to this shape. |
| `keypoints` | `List[cv2.KeyPoint]` | SIFT keypoints, loaded from `data/<id>/template_features.pkl` if cached. |
| `descriptors` | `np.ndarray` | SIFT descriptors. |
| `rois` | `List[ROI]` | Loaded from `data/<id>/config.json`. |

**Lifecycle**: Constructed once at `DocumentProcessor.__init__` via `template_loader.load_all_templates()`. SIFT features are auto-cached on first run.

---

### 2. ROI
**Source**: [vlm_pdf_recognizer/templates/__init__.py](../../vlm_pdf_recognizer/templates/__init__.py)

| Field | Type | Notes |
|---|---|---|
| `roi_id` | `str` | Must equal a `field_id` in `recognition/field_schema.py` for that template. |
| `description` | `str` | Human-readable label. |
| `bounding_box` | `Tuple[int, int, int, int]` | `(x1, y1, x2, y2)` in template image coordinates. |
| `visualization_color` | `Tuple[int, int, int]` | Default BGR color for the verification overlay. |

**Constraint**: All ROIs share their template's pixel grid — they are valid for the **aligned** document, not the raw scan.

---

### 3. ExtractedROI
**Source**: [vlm_pdf_recognizer/extraction/roi_extractor.py](../../vlm_pdf_recognizer/extraction/roi_extractor.py)

| Field | Type | Notes |
|---|---|---|
| `roi_id` | `str` | Same as the originating `ROI`. |
| `description` | `str` | |
| `bounding_box` | `Tuple[int,int,int,int]` | Same as the originating `ROI` (after warp, coordinates match template). |
| `visualization_color` | `Tuple[int,int,int]` | |
| `roi_image` | `np.ndarray` | BGR crop from the aligned image, passed downstream to Feature 002 / 004. |

---

### 4. FeatureMatch
**Source**: [vlm_pdf_recognizer/alignment/template_matcher.py](../../vlm_pdf_recognizer/alignment/template_matcher.py)

| Field | Type | Notes |
|---|---|---|
| `template_id` | `str` | |
| `total_matches` | `int` | Lowe-ratio-filtered matches. |
| `good_matches` | `List[cv2.DMatch]` | |
| `inliers` | `np.ndarray` | RANSAC inlier mask. |
| `inlier_count` | `int` | The voting metric. |
| `confidence` | `float` | `inlier_count / total_matches`. |
| `homography_matrix` | `np.ndarray` | 3×3, computed per template. |

---

### 5. TemplateMatchResult
**Source**: [vlm_pdf_recognizer/alignment/template_matcher.py](../../vlm_pdf_recognizer/alignment/template_matcher.py)

| Field | Type | Notes |
|---|---|---|
| `matched_template_id` | `str` | Winner of the inlier vote. |
| `matched_template` | `GoldenTemplate` | |
| `match_confidence` | `float` | Winner's confidence ratio. |
| `inlier_count` | `int` | Winner's inlier count. |
| `homography_matrix` | `np.ndarray` | Winner's homography (used for warp). |
| `all_matches` | `List[FeatureMatch]` | All candidates — useful for debugging. |

**Failure**: If `winner.inlier_count < 50`, `match_templates()` raises `UnknownDocumentError` instead of returning.

---

### 6. ProcessingResult
**Source**: [vlm_pdf_recognizer/pipeline.py](../../vlm_pdf_recognizer/pipeline.py)

| Field | Type | Notes |
|---|---|---|
| `input_path` | `str` | Original input file path. |
| `page_number` | `int` | 0 for images / single-page; ≥ 0 per PDF page. |
| `matched_template_id` | `str` | The winning template ID, or `"unknown"` / `"error"` on failure. |
| `confidence_score` | `int` | **Inlier count** (not a probability). |
| `processing_time_ms` | `float` | Wall-clock for this page. |
| `aligned_image` | `np.ndarray` | BGR, warped to template dimensions. |
| `visualization_image` | `np.ndarray` | Aligned image + ROI boxes (later overwritten by Feature 002 colour-coding when VLM is enabled). |
| `extracted_rois` | `List[ExtractedROI]` | Same order as `template.rois`. |
| `success` | `bool` | `True` only when matching, warping and crops all succeeded. |
| `error_message` | `Optional[str]` | Populated on failure. |

---

### 7. UnknownDocumentError
**Source**: [vlm_pdf_recognizer/alignment/template_matcher.py](../../vlm_pdf_recognizer/alignment/template_matcher.py)

Raised when the best template still has `< 50` RANSAC inliers. Error message includes per-template inlier breakdown for diagnostics.

---

## Data Flow

```
input/<date>/<case_id>/<file>                    (Path)
   │
   ▼
pdf_to_images()              ──►  List[BGR ndarray]   (one entry per page)
   │
   ▼
extract_features()           ──►  (keypoints, descriptors)   (doc: ≤5000 features)
   │
   ▼
match_templates()            ──►  TemplateMatchResult
   │                              └─ raises UnknownDocumentError if inliers < 50
   ▼
align_document_to_template() ──►  aligned_image           (warpPerspective)
   │
   ▼
extract_rois()               ──►  List[ExtractedROI]
   │
   ▼
draw_roi_boxes()             ──►  visualization_image
   │
   ▼
ProcessingResult
   │
   ▼  (downstream — Feature 002/004)
VLMRecognizer.process_document()
   │
   ▼
DocumentRecognitionOutput  (see specs/002 data-model)
```

---

## File-System Mapping

```
data/
├── contractor_1/
│   ├── config.json              # ROI coordinates (generated by update_configs.py)
│   ├── template_features.pkl    # Cached SIFT keypoints + descriptors
│   └── blank_rois/<field>.png   # Blank reference for Feature 004 AIP
├── contractor_2/
│   └── … (same structure)
└── enterprise_1/
    └── … (same structure)

templates/
├── images/<template_id>.jpg     # Golden template image
└── location/<template_id>.json  # LabelMe annotation (source of truth for ROIs)

input/
└── <date>/<case_id>/<file>      # Required two-level nesting

output/                          # Mirrors input layout per date/case
└── <date>/
    ├── VLM_results.json
    ├── vlm_recognition_results.csv
    ├── result_log.md
    └── <case_id>/
        ├── <doc>_visualization.png
        ├── metadata/<doc>_metadata.json
        ├── rois/<doc>_roi_<field>.png
        └── processed_rois/<doc>_roi_<field>_processed.png
```

---

## Template Config Schema (`data/<id>/config.json`)

The actual schema is whatever `update_configs.py` produces from LabelMe annotations. Conceptually:

```json
{
  "template_id": "contractor_1",
  "image_shape": [3508, 2480, 3],
  "rois": [
    {
      "roi_id": "VX1",
      "description": "Disagreement checkbox",
      "bounding_box": [200, 150, 800, 250],
      "visualization_color": [0, 255, 0]
    },
    ...
  ]
}
```

**Authoritative source**: regenerate `config.json` whenever `templates/location/<id>.json` (LabelMe) changes — never hand-edit `config.json`.

---

## Feature Cache (`data/<id>/template_features.pkl`)

```python
{
  "keypoints": [(pt, size, angle, response, octave, class_id), ...],
  "descriptors": np.ndarray,     # N x 128 (SIFT)
  "image_shape": (H, W, 3),
  "template_id": str,
  "created_at": ISO-8601 timestamp,
}
```

Cache invalidation is based on file modification time vs. the source template image — see `template_cache.py`.

---

## Errors & Recovery

| Error | Where | Recovery |
|---|---|---|
| `UnknownDocumentError` (< 50 inliers) | `match_templates()` | Caught in `DocumentProcessor.process_image()` → returns `ProcessingResult(success=False, matched_template_id="unknown")`. |
| `FileNotFoundError` | `DocumentProcessor.process_file()` | Re-raised to caller (`main.py`) — counted toward batch failure stats but does not abort the batch. |
| `cv2.error` during SIFT / FLANN / warp | `DocumentProcessor.process_image()` | Caught → `ProcessingResult(success=False, matched_template_id="error", error_message=str(e))`. |
| PDF load failure | `pdf_to_images()` | Surfaces as a generic exception → batch continues with next file. |

---

## What's Intentionally Absent From This Stage

- **No preprocessing of pixels** (watermark removal, binarisation) — removed.
- **No content presence detection** — owned by Feature 004 AIP downstream.
- **No text recognition** — owned by Feature 002 VLM downstream.
- **No validation logic** — owned by Feature 002 `calculate_results_status`.
- **No batch-job entity** — `main.py` is the orchestrator; per-batch state lives in local variables / log lines.
