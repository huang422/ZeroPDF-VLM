# Implementation Plan: VLM-Assisted ROI Content Detection with Image Preprocessing

**Branch**: `004-vlm-roi-preprocessing` | **Date**: 2025-12-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-vlm-roi-preprocessing/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a 6-step image preprocessing pipeline for ROI content detection that compares extracted document ROIs against blank template ROIs using template difference, morphological operations, and connected components analysis. The pipeline provides deterministic has_content detection (True/False) for signature boxes, stamp areas, and text fields, replacing the previous SIFT feature matching approach with more reliable pixel-based analysis. Preprocessing results are used as primary input for document validation logic while maintaining full VLM integration for content extraction.

## Technical Context

**Language/Version**: Python 3.9+ (existing vlmcv environment)
**Primary Dependencies**:
- OpenCV (cv2) 4.x - Image preprocessing, morphological operations, connected components
- NumPy - Array operations and pixel calculations
- PyTorch 2.0+ - Existing VLM model execution
- Transformers 4.52.1+ (HuggingFace) - Existing VLM infrastructure
- Pillow (PIL) - Existing image format conversion

**Storage**: File-based
- Blank template ROIs: data/{template_id}/blank_rois/{field_id}.png
- Processed ROI debug images: output/processed_rois/{document_name}/{field_id}/
- Configuration: vlm_pdf_recognizer/recognition/config.py (Python module)
- Output: JSON and CSV files in output/ directory

**Testing**: pytest (existing test infrastructure)
- Unit tests: tests/unit/test_roi_preprocessor.py (new)
- Integration tests: tests/integration/test_preprocessing_pipeline.py (new)
- Existing tests: Ensure zero regression on VLM, checkbox, template matching

**Target Platform**: Linux (primary), cross-platform Python
**Project Type**: Single Python application with CLI interface (main.py)

**Performance Goals**:
- Preprocessing pipeline: <100ms per ROI (CPU-based OpenCV)
- Total overhead: <10-20% increase to existing pipeline time
- Blank ROI generation: <10 seconds for all templates

**Constraints**:
- Must preserve all existing functionality (zero regression)
- Must integrate with existing VLM pipeline without breaking backward compatibility
- Must save intermediate images for debugging without significant performance impact
- Configurable thresholds via config.py (no hardcoded values)

**Scale/Scope**:
- 3 templates (contractor_1, contractor_2, enterprise_1)
- 13-14 ROI fields per template
- Batch processing of multiple documents
- 6 intermediate images per ROI per document (for debugging)

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
├── preprocessing/           # Existing module
│   └── pdf_converter.py
├── alignment/              # Existing module (template matching, ROI extraction)
│   ├── feature_extractor.py
│   ├── template_matcher.py
│   ├── geometric_corrector.py
│   ├── blank_roi_cache.py
│   └── roi_comparator.py   # MODIFIED: Remove SIFT comparison logic
├── extraction/             # Existing module
│   └── roi_extractor.py
├── recognition/            # Existing module - MAJOR MODIFICATIONS
│   ├── __init__.py
│   ├── config.py          # MODIFIED: Add preprocessing thresholds
│   ├── field_schema.py    # Existing
│   ├── vlm_loader.py      # Existing
│   ├── vlm_recognizer.py  # MODIFIED: Integrate preprocessing pipeline
│   └── roi_preprocessor.py # NEW: 6-step preprocessing pipeline
├── templates/             # Existing module
│   ├── template_cache.py
│   └── template_loader.py
├── pipeline.py            # MODIFIED: Load blank template ROI cache
└── output.py              # MODIFIED: Export preprocessing results

data/
├── {template_id}/
│   ├── blank_rois/        # NEW: Blank template ROI images
│   │   ├── {field_id}.png
│   │   └── ...
│   ├── template.png       # Existing
│   └── config.json        # Existing

output/
├── processed_rois/        # NEW: Preprocessing debug images
│   └── {document_name}/
│       └── {field_id}/
│           ├── 01_saturation.png
│           ├── 02_difference.png
│           ├── 03_hline_removed.png
│           ├── 04_vline_removed.png
│           ├── 05_noise_removed.png
│           ├── 06_final_binary.png
│           └── metadata.json
├── VLM_results.json       # MODIFIED: Add preprocessing columns
└── vlm_recognition_results.csv  # MODIFIED: Add preprocessing columns

tests/
├── unit/
│   ├── test_roi_preprocessor.py  # NEW: Unit tests for preprocessing
│   ├── test_vlm_recognizer.py   # MODIFIED: Update for preprocessing
│   └── ...
└── integration/
    ├── test_preprocessing_pipeline.py  # NEW: End-to-end tests
    └── ...

update_configs.py          # MODIFIED: Generate blank template ROIs
main.py                    # Existing (no changes needed)
```

**Structure Decision**: Single Python application structure (Option 1). The feature integrates into the existing vlm_pdf_recognizer package by:
1. Adding new roi_preprocessor.py module for the 6-step pipeline
2. Modifying vlm_recognizer.py to call preprocessing before VLM inference
3. Extending update_configs.py to generate blank template ROI images
4. Updating config.py with new preprocessing threshold parameters
5. Modifying output.py to export preprocessing results alongside VLM results

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: ✅ PASS (No constitution file found - proceeding with feature implementation)

**Notes**:
- No `.specify/memory/constitution.md` constraints defined for this project
- Following existing codebase patterns and architecture
- Maintaining backward compatibility with all existing features
- Zero regression requirement enforced via testing

## Complexity Tracking

*No violations - Constitution Check passed. This section is not applicable.*
