# Feature Specification: Document Template Alignment & ROI Extraction

**Feature Branch**: `001-document-template-alignment`
**Created**: 2025-12-23
**Last Aligned With Code**: 2026-05-26
**Status**: Implemented — this spec reflects current production code in `vlm_pdf_recognizer/`
**Input**: Local, zero-shot document processing system for Traditional Chinese scanned PDF documents with template-based classification, geometric alignment, and ROI extraction. This feature delivers the preprocessing half of the pipeline; recognition is covered by Features 002 / 004.

---

## Scope Recap (Current Implementation)

This feature owns the path from **raw scan → aligned image with cropped ROI images**:

```
input PDF / image
  → PyMuPDF page conversion (preprocessing/pdf_converter.py)
  → SIFT feature extraction (alignment/feature_extractor.py, doc cap 5000 features)
  → FLANN matching + Lowe ratio + RANSAC voting (alignment/template_matcher.py)
  → Perspective warp via homography (alignment/geometric_corrector.py)
  → ROI crop from aligned image (extraction/roi_extractor.py)
  → ProcessingResult with extracted_rois + visualization image
```

Downstream stages (AIP content detection, VLM text recognition, validation, case aggregation) are owned by Features 002/004 and `vlm_pdf_recognizer/recognition/`.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Process Single Scanned PDF (Priority: P1)

A user has a scanned PDF or image of a structured Traditional Chinese form (e.g. authorization letter). They need the system to (a) decide which template it matches, (b) align it to the template's canonical pixel grid, and (c) crop out every pre-defined ROI region for downstream content detection.

**Independent Test**: Drop one PDF into `input/<date>/<case_id>/`, run `python main.py --disable-vlm`, and confirm:
- `output/<date>/<case_id>/<doc>_visualization.png` shows the aligned image with all ROI boxes drawn.
- `output/<date>/<case_id>/metadata/<doc>_metadata.json` reports `matched_template_id`, `confidence_score` (RANSAC inlier count), and per-ROI bounding boxes.

**Acceptance Scenarios**:

1. **Given** a scanned `contractor_1` form with mild rotation (≤ 15°), **When** the user runs the pipeline, **Then** the system selects `contractor_1`, warps to template dimensions pixel-for-pixel, and draws all 13 ROI bounding boxes.
2. **Given** a multi-page PDF (3 pages, different template per page), **When** the pipeline runs, **Then** each page becomes its own `ProcessingResult` with an independent template match.
3. **Given** an image-format input (`.png`, `.jpg`), **When** the pipeline runs, **Then** behaviour matches PDF input (single-page processing path).

---

### User Story 2 — Reject Documents That Don't Match Any Template (Priority: P1)

A user submits a document that does not correspond to any registered template (e.g. a passport scan, or a contractor form with a totally different layout). The system must clearly fail rather than warp arbitrary content and produce misleading downstream results.

**Independent Test**: Place an unrelated document in `input/<date>/<case_id>/`, run the pipeline, and confirm `ProcessingResult.success=False`, `matched_template_id="unknown"`, and `error_message` includes per-template inlier breakdowns.

**Acceptance Scenarios**:

1. **Given** a document whose best template has fewer than **50 RANSAC inliers**, **When** matching runs, **Then** `UnknownDocumentError` is raised inside `match_templates()` and converted to a failed `ProcessingResult` with `matched_template_id="unknown"`.
2. **Given** a heavily blurred / low-DPI scan, **When** SIFT extracts insufficient descriptors, **Then** matching reports zero inliers for all templates and the document is rejected.
3. **Given** ambiguous matches (multiple templates near threshold), **When** voting runs, **Then** the system picks the template with the **maximum inlier count** (winner-takes-all, no margin enforcement).

---

### User Story 3 — Batch Process Nested Case Directories (Priority: P1)

The production input is organised as `input/<date>/<case_id>/*.pdf` (multiple PDFs per case, multiple cases per date). The system walks this nested structure, mirrors the layout under `output/`, and continues on per-file failures.

**Independent Test**: Populate `input/2026-05-25/case_a/` with three PDFs and `input/2026-05-25/case_b/` with two PDFs. Run the pipeline. Confirm `output/2026-05-25/case_a/` and `output/2026-05-25/case_b/` each contain visualisations and metadata; failures appear in the per-date `VLM_results.json`.

**Acceptance Scenarios**:

1. **Given** `input/<date>/<case_id>/*.pdf` with mixed templates, **When** `main.py` runs, **Then** outputs land in the mirrored `output/<date>/<case_id>/` paths with one visualisation per page.
2. **Given** one file in the batch fails (e.g. unreadable PDF), **When** processing continues, **Then** other files complete normally and the failure is recorded with `success=False`.
3. **Given** a batch spanning multiple dates, **When** processing finishes, **Then** each date directory gets its own `VLM_results.json` (preprocessing-only stats when `--disable-vlm`).

---

### Edge Cases (Behaviour in Code Today)

| Scenario | Current Behaviour |
|---|---|
| Document upside-down / rotated 180° | Homography handles in-plane rotation; if rotation is 180° **and** features remain symmetric enough, RANSAC will fail and `UnknownDocumentError` fires. No explicit orientation pre-flip. |
| Photographed (perspective distortion) | Full 8-DOF homography in `findHomography` can correct moderate perspective. Extreme distortion → low inliers → reject. |
| Skew > 30° | Likely rejected (insufficient inliers). No documented success guarantee. |
| Partially cropped page | If enough of the page is visible to clear 50 inliers, alignment succeeds; cropped regions outside the warped frame become whitespace. |
| Multi-page PDF | Each page processed independently (`pdf_to_images` returns one BGR array per page); each gets its own `ProcessingResult`. |
| Failed PDF load | Caught in `process_file` → propagated as `ProcessingResult(success=False, error_message=...)`. |
| Watermarks / coloured backgrounds | **No watermark removal step** — SIFT is robust to translucent watermarks, so the pipeline feeds the original BGR image directly to feature extraction. |

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support three document template types: `enterprise_1`, `contractor_1`, `contractor_2`. Template list lives in `vlm_pdf_recognizer/templates/template_loader.load_all_templates()`.
- **FR-002**: System MUST load golden templates and their cached SIFT features from `data/<template_id>/` at startup via `DocumentProcessor._load_templates()`.
- **FR-003**: System MUST load ROI coordinate configuration from `data/<template_id>/config.json` (auto-generated by `update_configs.py` from LabelMe annotations in `templates/location/`).
- **FR-004**: System MUST process the original BGR image directly through SIFT — **no watermark removal, colour thresholding, or binarisation is applied before feature extraction.** (Removed because SIFT is robust to translucent watermarks and removing them was reducing feature richness.)
- **FR-005**: System MUST extract SIFT keypoints and descriptors from input images with a cap of **5000 features for documents** (templates use unlimited features to maximise matching accuracy).
- **FR-006**: System MUST match document descriptors against all loaded templates using **FLANN-based matcher (KDTree, trees=5, checks=50)** with **Lowe's ratio test (ratio = 0.7)**.
- **FR-007**: System MUST compute homography per template using **`cv2.findHomography` with RANSAC (reprojection threshold = 5.0 pixels)**, returning the inlier mask.
- **FR-008**: System MUST select the template with the **maximum RANSAC inlier count** (voting / winner-takes-all). No tie-break logic — `max(..., key=lambda m: m.inlier_count)` decides.
- **FR-009**: System MUST raise `UnknownDocumentError` when the winning template has **fewer than 50 RANSAC inliers**, and convert that exception into a `ProcessingResult` with `matched_template_id="unknown"` and `success=False`.
- **FR-010**: System MUST warp the original colour image to the matched template's pixel dimensions using `cv2.warpPerspective` (the aligned image becomes the canonical reference frame for ROI crop and AIP comparison).
- **FR-011**: System MUST extract every ROI defined in the template config from the aligned image, producing `ExtractedROI` instances with `roi_id`, `bounding_box`, `description`, `visualization_color`, and `roi_image` (BGR crop).
- **FR-012**: System MUST draw all ROI bounding boxes on the aligned image to produce a verification visualisation. (When VLM is enabled, this visualisation is later **overwritten** by Feature 002's colour-coded version.)
- **FR-013**: System MUST persist the visualisation as `<base>_visualization.png` and per-document metadata as `metadata/<base>_metadata.json` under the mirrored case output directory.
- **FR-014**: System MUST cache pre-computed SIFT keypoints + descriptors for each template at `data/<template_id>/template_features.pkl` so subsequent runs skip template feature extraction.
- **FR-015**: System MUST accept input from a **nested directory layout** `input/<date>/<case_id>/<file>` and mirror that layout under `output/`. Supported extensions: `.pdf`, `.jpg`, `.jpeg`, `.png`.
- **FR-016**: System MUST process multi-page PDFs (PyMuPDF / `fitz`) one page at a time, emitting one `ProcessingResult` per page with `page_number` set.
- **FR-017**: System MUST continue processing remaining files when a single file errors, logging the failure to the batch summary rather than aborting.
- **FR-018**: System MUST run end-to-end on CPU (SIFT, FLANN, warp, ROI crop — all CPU-bound OpenCV). GPU acceleration applies only to the downstream VLM stage (Feature 002), not to alignment.
- **FR-019**: System SHOULD be invokable with `--disable-vlm` to run alignment + ROI extraction only (no Feature 002/004), for offline debugging of the alignment stage.

### Key Entities

| Entity | Module | Notes |
|---|---|---|
| `GoldenTemplate` | `vlm_pdf_recognizer/templates/__init__.py` | Holds `template_id`, `image_shape`, `keypoints`, `descriptors`, list of `ROI` definitions. |
| `ROI` | `vlm_pdf_recognizer/templates/__init__.py` | Bounding box + metadata (`roi_id`, `description`, `visualization_color`). |
| `ExtractedROI` | `vlm_pdf_recognizer/extraction/roi_extractor.py` | The cropped BGR pixel array plus the originating ROI metadata. |
| `FeatureMatch` | `vlm_pdf_recognizer/alignment/template_matcher.py` | Per-template matching result (matches, inliers, homography). |
| `TemplateMatchResult` | same | Winning template + all candidates for debugging. |
| `ProcessingResult` | `vlm_pdf_recognizer/pipeline.py` | The dataclass returned per page: matched template, inlier count, aligned image, visualisation, list of `ExtractedROI`, success flag. |
| `UnknownDocumentError` | `vlm_pdf_recognizer/alignment/template_matcher.py` | Raised when no template clears the 50-inlier floor. |

---

## Success Criteria *(mandatory)*

These are operational targets observed on RTX 4080 Laptop / Ubuntu (the development reference machine — see README "Performance" table for source of truth):

- **SC-001**: For documents in scope (correctly photographed scans of the three template types), the correct template wins by inlier count (no quantitative accuracy benchmark currently tracked in tests).
- **SC-002**: End-to-end per-page time for the alignment stage is **≤ 1 second** (PDF conversion < 100 ms, SIFT 100–300 ms, FLANN+RANSAC 200–500 ms, warp 50–100 ms, ROI crop < 100 ms).
- **SC-003**: When `--disable-vlm` is set, no VLM dependencies are required for the alignment pipeline to run.
- **SC-004**: Template feature cache (`template_features.pkl`) reduces subsequent startup time relative to recomputing SIFT on each run.
- **SC-005**: Documents below the 50-inlier threshold are flagged with `matched_template_id="unknown"` and `success=False` rather than aligned to a random template.
- **SC-006**: A failing file does not abort the batch — remaining files complete and the failure is surfaced in the per-date `VLM_results.json` and stdout summary.

---

## Assumptions

1. **Input layout**: Production input always follows `input/<date>/<case_id>/<file>`. `main.py:scan_nested_input()` ignores any file outside that two-level nesting.
2. **Template stability**: The three template types are fixed. Adding a new template requires (a) a new entry in `load_all_templates()`, (b) an annotated `templates/location/<id>.json`, and (c) re-running `update_configs.py`.
3. **Language scope**: This feature is purely visual — it does **not** read or interpret text content. Text recognition is owned by Feature 002 (VLM) and Feature 004 (AIP detects presence, not text).
4. **DPI & quality**: Scans are assumed to be at a resolution that yields ≥ 50 inliers against the template. No automatic upscaling.
5. **Template assets**: `templates/images/<id>.jpg` and `templates/location/<id>.json` exist and are in sync with `data/<template_id>/config.json` (regenerate via `update_configs.py` whenever annotations change).
6. **Filesystem**: Read access to `data/` and `input/`, write access to `output/`.

---

## Out of Scope

- OCR / text extraction → owned by Feature 002.
- AIP / pixel-level content presence detection → owned by Feature 004.
- Validation logic (VX1 priority, date OR, AND of remaining fields, VX2 must be checked) → owned by Feature 002 (`vlm_recognizer.calculate_results_status`).
- Case-level aggregation (three-template completeness check) → owned by Feature 002 (`output.py:_aggregate_case_results`).
- Watermark removal / colour thresholding / binarisation (removed from the pipeline).
- Real-time / streaming input.
- Web UI or cloud deployment — CLI batch only.
- Adding new template types at runtime (configuration is offline via `update_configs.py`).

---

## Dependencies

- **OpenCV ≥ 4.8** — SIFT, FLANN, RANSAC, `warpPerspective`, image I/O.
- **PyMuPDF (`fitz`) ≥ 1.23** — PDF → image.
- **NumPy ≥ 1.24** — array operations.
- **Pillow ≥ 10.0** — auxiliary image I/O.
- **`update_configs.py`** must have been run at least once to populate `data/<template_id>/config.json`, `blank_rois/`, and `template_features.pkl`.

---

## Notes for Spec-vs-Code Reviewers

- The `--disable-vlm` flag in `main.py` is the cleanest way to exercise this feature in isolation.
- The visualisation written by this feature is intentionally **overwritten** by `output.save_vlm_visualization()` when VLM is enabled — that is by design, not a bug.
- "Confidence score" in `ProcessingResult.confidence_score` is the **RANSAC inlier count**, not a normalised probability.
