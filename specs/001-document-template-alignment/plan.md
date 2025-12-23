# Implementation Plan: Document Template Alignment & ROI Extraction

**Branch**: `001-document-template-alignment` | **Date**: 2025-12-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-document-template-alignment/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a local, zero-shot document processing system that automatically classifies scanned PDF documents (Traditional Chinese) into one of three template types (enterprise_1, contractor_1, contractor_2), removes watermarks via HSV thresholding, aligns documents to golden templates using SIFT feature matching and homography transformation, and extracts predefined ROI regions with bounding box visualization. System operates on CPU or GPU (4GB VRAM min) using OpenCV for all computer vision operations.

## Technical Context

**Language/Version**: Python 3.9+ (vlmcv environment)
**Primary Dependencies**:
- OpenCV (cv2) 4.x - SIFT feature extraction, feature matching, homography, perspective transforms
- NumPy - Array operations and image manipulation
- PyMuPDF (fitz) - PDF to image conversion with dimension preservation

**Storage**: File-based - Golden templates + JSON configs in data/, outputs to output/
**Testing**: pytest with opencv-python testing utilities
**Target Platform**: Linux (primary), CPU-only or CUDA-enabled GPU (4GB+ VRAM)
**Project Type**: Single project (command-line tool/library)
**Performance Goals**:
- CPU mode: <10 seconds per document
- GPU mode: <3 seconds per document
- Batch: 100 documents with <3% failure rate

**Constraints**:
- 4GB VRAM minimum (GPU mode) or 8GB RAM (CPU mode)
- Zero-shot (no training) - template matching only
- Preserve original image dimensions through PDF conversion
- SIFT feature cache persistence for golden templates

**Scale/Scope**:
- 3 document templates (enterprise_1, contractor_1, contractor_2)
- Single document + batch processing modes
- Multi-page PDF support (each page processed independently)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: ⚠️ No project constitution defined - using default best practices

Since the constitution file is a template placeholder, applying standard software engineering principles:

✅ **Modularity**: Design as library with CLI interface (passes)
✅ **Testability**: pytest-based testing planned (passes)
✅ **Simplicity**: Single-purpose tool, no over-engineering (passes)
✅ **Documentation**: Spec + plan + quickstart planned (passes)

No violations detected. Will re-check after Phase 1 design.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
vlm_pdf_recognizer/
├── __init__.py
├── preprocessing/
│   ├── __init__.py
│   ├── watermark_removal.py    # HSV thresholding, binarization
│   └── pdf_converter.py         # PDF to image conversion (multi-page)
├── alignment/
│   ├── __init__.py
│   ├── feature_extractor.py    # SIFT feature extraction
│   ├── template_matcher.py     # Feature matching, voting, classification
│   └── geometric_corrector.py  # Homography, perspective warp
├── extraction/
│   ├── __init__.py
│   └── roi_extractor.py        # ROI extraction, bounding box overlay
├── templates/
│   ├── __init__.py
│   ├── template_loader.py      # Load golden templates + JSON configs
│   └── template_cache.py       # SIFT feature caching
├── pipeline.py                  # Main processing pipeline orchestration
├── batch_processor.py           # Batch processing with error handling
└── cli.py                       # Command-line interface

tests/
├── unit/
│   ├── test_watermark_removal.py
│   ├── test_feature_extraction.py
│   ├── test_template_matching.py
│   ├── test_geometric_correction.py
│   └── test_roi_extraction.py
├── integration/
│   ├── test_pipeline_single.py
│   ├── test_pipeline_batch.py
│   └── test_pdf_conversion.py
└── fixtures/
    ├── sample_docs/             # Test documents
    └── golden_templates/        # Test templates

data/                            # Golden templates + configs (runtime)
├── enterprise_1/
│   ├── template.png
│   ├── template_features.pkl   # Cached SIFT features
│   └── config.json             # ROI coordinates
├── contractor_1/
│   └── [same structure]
└── contractor_2/
    └── [same structure]

output/                          # Processed outputs (runtime)
└── [generated at runtime]
```

**Structure Decision**: Single project layout chosen as this is a standalone document processing tool. Core processing modules (preprocessing, alignment, extraction) separated by functional responsibility. Template management isolated in dedicated module. CLI provides entry point for both single and batch processing modes.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected. System follows simplicity principles:
- Single-purpose tool (document alignment only)
- File-based storage (no database complexity)
- Modular design with clear separation of concerns
- Minimal dependencies (OpenCV, NumPy, PyMuPDF)

---

## Phase 0: Research (Completed)

✅ **research.md** created with technical decisions:
- PDF Library: PyMuPDF (fitz) selected
- SIFT Parameters: Optimized for document matching
- Watermark Removal: HSV thresholding strategy
- Homography: RANSAC with 5.0 pixel threshold
- ROI Config: JSON format defined
- Feature Cache: Pickle serialization approach

**All NEEDS CLARIFICATION items resolved.**

---

## Phase 1: Design (Completed)

✅ **data-model.md** created defining 8 core entities:
- GoldenTemplate, ROI, InputDocument, DocumentPage
- FeatureMatch, ProcessingResult, ExtractedROI, BatchProcessingJob
- Data flow and state transitions documented
- File structure mapping defined

✅ **quickstart.md** created with:
- Installation instructions
- Setup guide for golden templates
- Basic usage examples (single + batch processing)
- Troubleshooting guide

✅ **Agent context updated** (CLAUDE.md):
- Language: Python 3.9+ (vlmcv)
- Storage: File-based
- Project type: Single CLI tool

---

## Constitution Re-Check (Post-Design)

**Status**: ✅ PASSED - No violations introduced

**Design Review**:
✅ **Modularity**: 6 functional modules (preprocessing, alignment, extraction, templates, pipeline, batch)
✅ **Testability**: Unit + integration test structure defined
✅ **Simplicity**: No unnecessary abstractions, direct file I/O
✅ **Documentation**: Complete spec + research + data model + quickstart

**Complexity Justified**:
- Feature caching: Required for 40% performance improvement (SC-009)
- RANSAC algorithm: Required for robustness against handwriting noise (FR-010)
- Multi-stage pipeline: Necessary for watermark removal → alignment → extraction workflow

---

## Phase 2: Next Steps

To generate actionable tasks, run:

```bash
/speckit.tasks
```

This will create `tasks.md` with dependency-ordered implementation tasks based on the plan and design artifacts.
