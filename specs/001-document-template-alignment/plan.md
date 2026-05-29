# Implementation Plan: Document Template Alignment & ROI Extraction

**Branch**: `001-document-template-alignment` | **Date**: 2025-12-23 | **Spec**: [spec.md](spec.md)
**Last Aligned With Code**: 2026-05-26
**Status**: Implemented — this plan now reflects the production design rather than the original draft. Drifts from the original draft are noted inline.

## Summary

Build the alignment-stage half of a local, zero-shot document-recognition pipeline for Traditional Chinese forms. The feature ingests scanned PDFs / images, classifies each page to one of three templates (`contractor_1`, `contractor_2`, `enterprise_1`) using SIFT + FLANN + RANSAC voting, warps to the template's canonical pixel grid via homography, and crops every pre-defined ROI for the downstream AIP + VLM stages. Everything runs on CPU; no model training is involved.

> **Drift from original draft**: the draft listed HSV watermark removal as a step. That step was deleted because SIFT is robust to translucent watermarks and removing them was reducing feature richness — production code feeds the original BGR image directly to SIFT (see `pipeline.py:97-103`).

## Technical Context

| Item | Value |
|---|---|
| Language | Python 3.9+ |
| Image processing | OpenCV ≥ 4.8 (SIFT, FLANN, RANSAC, `warpPerspective`) |
| PDF | PyMuPDF / `fitz` ≥ 1.23 |
| Array math | NumPy ≥ 1.24 |
| Storage | Pure filesystem — `data/`, `input/`, `output/`. No DB. |
| Tests | pytest, fixtures in `tests/` |
| Hardware | CPU-bound. GPU only matters for the downstream VLM stage (Feature 002), **not** this stage. |
| Throughput | 3–5 s end-to-end per page on RTX 4080 Laptop (alignment stage alone is < 1 s; VLM dominates). |

## Constitution Check

See [.specify/memory/constitution.md](../../.specify/memory/constitution.md). Compliance highlights:

- **Local-first**: All alignment runs on the user's machine. No network calls during this stage.
- **Zero training**: SIFT + homography only; no model is trained or fine-tuned.
- **Determinism on the alignment path**: RANSAC has a seed in practice via OpenCV defaults; the same input + template set yields the same template winner.
- **Graceful degradation**: A non-matching document becomes a logged `success=False` row, not a crash.

## Project Structure

### Documentation (this feature)
```text
specs/001-document-template-alignment/
├── spec.md          # Source of truth for behaviour
├── plan.md          # This file
├── research.md      # Original technical choices (SIFT params, RANSAC threshold)
├── data-model.md    # Dataclass reference
├── quickstart.md    # How to exercise this stage alone
├── tasks.md         # Historical implementation task list (all completed)
└── checklists/
```

### Source Code (current production layout — note differences from the original draft)
```text
vlm_pdf_recognizer/
├── __init__.py
├── pipeline.py                  # DocumentProcessor orchestrates this feature
├── preprocessing/
│   └── pdf_converter.py         # PyMuPDF page → BGR ndarray
├── alignment/
│   ├── feature_extractor.py     # SIFT (doc cap 5000, templates unlimited)
│   ├── template_matcher.py      # FLANN + Lowe + RANSAC voting; raises UnknownDocumentError
│   ├── geometric_corrector.py   # warpPerspective wrapper
│   └── blank_template_roi_cache.py   # cache used by Feature 004 — not by this feature
├── extraction/
│   └── roi_extractor.py         # Crop ROIs from aligned image; draw_roi_boxes
├── templates/
│   ├── __init__.py              # ROI + GoldenTemplate dataclasses
│   ├── template_loader.py       # Iterate templates from data/
│   └── template_cache.py        # SIFT keypoint/descriptor pickle cache
├── recognition/                 # OWNED BY FEATURES 002 / 004 — not this feature
└── output.py                    # OWNED BY FEATURES 002 / 004 — not this feature

main.py                          # Batch entry point (input/<date>/<case_id>/*.pdf walk)
```

**Notable deltas from the draft plan:**

| Draft said | Production is |
|---|---|
| `preprocessing/watermark_removal.py` | Doesn't exist (removed). |
| `batch_processor.py` | Folded into `main.py` (the orchestrator). |
| `cli.py` | Folded into `main.py`. |
| `data/<id>/template.png` | `templates/images/<id>.jpg`; the `data/` dir holds generated config + features, not the golden image. |
| Output `output/<timestamp>_<file>_pageN/` | `output/<date>/<case_id>/` mirroring the input layout. |

## Complexity Notes

- **Why feature cap = 5000 for documents** (FR-005 in spec): unlimited SIFT on full-resolution scans was slow and yielded diminishing returns; 5000 captures dominant document structure while keeping matching under ~500 ms.
- **Why minimum inlier = 50** (FR-009 in spec): empirically the lowest inlier count consistent with a correctly aligned warp. Below this, homography becomes noisy.
- **Why winner-takes-all (no margin enforcement)**: the three templates are sufficiently distinct visually that ties are rare; adding a margin rule would only reject borderline-correct documents.

## Phase 0 — Research (completed)

Decisions captured in [research.md](./research.md):
- PDF library: PyMuPDF.
- SIFT contrast/edge thresholds: defaults.
- FLANN: KDTree, trees=5, checks=50.
- Lowe ratio: 0.7. RANSAC reproj threshold: 5.0 px.
- Feature cache: pickle, invalidated by mtime.

Open items at the time → all now resolved by the implementation.

## Phase 1 — Design (completed)

Captured in [data-model.md](./data-model.md). The original 8-entity sketch was simplified during implementation:

- **Dropped** `InputDocument` and `DocumentPage` — they were just transient lists of ndarrays.
- **Dropped** `BatchProcessingJob` — `main.py` orchestrates without a dedicated entity.
- **Added** `BlankTemplateROICache` (lives here architecturally but is consumed by Feature 004).

## Phase 2 — Tasks

See [tasks.md](./tasks.md). All implementation tasks are complete. New work for this stage now flows through the standard PR review process, not through `/speckit.tasks` — see the project-level `DEVELOPMENT_WORKFLOW.md` for the current workflow.

## Maintenance Notes

When changing this stage, prefer touching:
- **Matching parameters** (Lowe ratio, RANSAC threshold, inlier floor): edit `template_matcher.py` and update FR-006 to FR-009 in `spec.md`.
- **Feature cap**: edit `feature_extractor.py` and update FR-005.
- **Input layout**: edit both `main.py:scan_nested_input()` and FR-015.

Any change that affects observable output (e.g. visualisation colours, ROI metadata fields, error messages downstream of `UnknownDocumentError`) must also update `data-model.md` and the matching FR in `spec.md`.
