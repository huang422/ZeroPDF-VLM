# Implementation Plan: VLM Auxiliary ROI Comparison System

**Branch**: `003-vlm-auxiliary-roi-comparison` | **Date**: 2025-12-30 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-vlm-auxiliary-roi-comparison/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature adds a pixel-based auxiliary ROI comparison system that works alongside VLM inference to improve content detection accuracy. The system extracts SIFT features from blank template ROIs during configuration, then compares document ROIs against these blank features during processing. High similarity (≥60% match) indicates an unfilled field, while low similarity indicates potential content. The auxiliary comparison result takes priority in validation logic, reducing false positives from VLM hallucination while maintaining transparency by preserving both auxiliary and VLM results in outputs.

## Technical Context

**Language/Version**: Python 3.9+ (vlmcv environment)
**Primary Dependencies**:
- OpenCV 4.x (SIFT feature extraction, FLANN matching, existing in project)
- NumPy (array operations, existing in project)
- PyTorch 2.0+ (existing for VLM, used for auxiliary comparison integration)
- Existing alignment modules (vlm_pdf_recognizer.alignment.feature_extractor, template_matcher)

**Storage**: File-based
- Input: Blank template images from templates/images/
- Output: Blank ROI features stored as .npz files in data/{template_id}/blank_roi_features.npz
- Output: Blank ROI images in data/{template_id}/rois/blank_{field_id}.png
- Existing: Template SIFT features in data/{template_id}/features.npz

**Testing**: pytest
- Unit tests for ROI comparison logic
- Unit tests for blank feature extraction
- Integration tests for end-to-end auxiliary comparison pipeline
- Backward compatibility tests to ensure existing VLM-only mode still works

**Target Platform**: Linux (primary), cross-platform Python (CPU and GPU support)

**Project Type**: Single project (Python package with CLI)

**Performance Goals**:
- Auxiliary ROI comparison: <50ms per ROI (comparable to template matching)
- Configuration script: Process all templates in <30 seconds
- Zero regression: Existing pipeline performance unchanged when auxiliary disabled

**Constraints**:
- Similarity threshold: 0.6 (60% feature match ratio) for filled vs unfilled determination
- Memory: Blank ROI features cached in memory (<5MB per template × 3 templates = <15MB)
- Backward compatibility: Existing outputs and VLM inference must remain functional

**Scale/Scope**:
- 3 document templates (contractor_1: 13 fields, contractor_2: 2 fields, enterprise_1: 14 fields)
- Maximum 14 ROIs per document to compare against blank features
- Batch processing: Typical workload 100-1000 documents per run

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: ✅ PASS (No constitution file defined - using default best practices)

The project does not have a populated constitution.md file. Following standard Python project best practices:
- Single project structure (existing vlm_pdf_recognizer package)
- Test-driven development approach with pytest
- Clear separation of concerns (alignment, recognition, extraction modules)
- File-based configuration and storage
- No architectural violations identified

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
├── alignment/
│   ├── feature_extractor.py      # Existing SIFT feature extraction (reuse)
│   ├── template_matcher.py       # Existing FLANN matching & RANSAC (reuse)
│   ├── roi_comparator.py         # NEW: Auxiliary ROI comparison logic
│   └── blank_roi_cache.py        # NEW: Blank ROI feature loader
├── recognition/
│   ├── vlm_recognizer.py         # MODIFY: Add auxiliary_has_content integration
│   ├── field_schema.py           # EXISTING: Field type definitions (reference)
│   └── vlm_loader.py             # EXISTING: VLM model loading (no changes)
├── extraction/
│   └── roi_extractor.py          # EXISTING: ROI extraction (no changes)
├── templates/
│   └── template_loader.py        # MODIFY: Load blank ROI features cache
└── output.py                      # MODIFY: Add auxiliary columns to outputs

update_configs.py                  # MODIFY: Extract blank ROI features
main.py                            # MINIMAL: Pass blank features to processor

data/{template_id}/
├── features.npz                   # EXISTING: Template SIFT features
├── blank_roi_features.npz         # NEW: Blank ROI SIFT features
└── rois/
    └── blank_{field_id}.png       # NEW: Extracted blank ROI images

tests/
├── unit/
│   ├── test_roi_comparator.py    # NEW: Test auxiliary comparison logic
│   ├── test_blank_roi_cache.py   # NEW: Test blank feature loading
│   └── test_vlm_recognizer.py    # MODIFY: Test auxiliary integration
└── integration/
    ├── test_vlm_pipeline.py       # MODIFY: Test end-to-end with auxiliary
    └── test_backward_compat.py    # NEW: Test VLM-only fallback mode
```

**Structure Decision**: Single project structure following existing vlm_pdf_recognizer package layout. New auxiliary comparison functionality added to the `alignment/` module (where SIFT feature matching already exists), with integration points in `recognition/` for VLM coordination and `templates/` for blank feature loading. This maintains clear separation of concerns while reusing existing infrastructure.

## Complexity Tracking

**Status**: No violations - no complexity tracking needed.

This feature follows existing architectural patterns and reuses infrastructure already in place for template matching. No new complexity introduced beyond the core feature requirements.
