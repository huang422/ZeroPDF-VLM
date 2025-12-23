# Tasks: Document Template Alignment & ROI Extraction

**Input**: Design documents from `/specs/001-document-template-alignment/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

**Tests**: Not requested in specification - focusing on core implementation

**Organization**: Tasks grouped by user story to enable independent implementation and testing

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Project structure: Single Python project with `vlm_pdf_recognizer/` source directory and `tests/` directory at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan (vlm_pdf_recognizer/, data/, output/, tests/)
- [X] T002 Initialize Python project with requirements.txt (opencv-python, numpy, PyMuPDF)
- [X] T003 [P] Create README.md with installation and basic usage instructions
- [X] T004 [P] Create .gitignore for Python project (*.pyc, __pycache__/, output/, *.pkl)
- [X] T005 [P] Create data/ directory structure (enterprise_1/, contractor_1/, contractor_2/)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Create template configuration JSON schema in data/enterprise_1/config.json (example with ROI definitions)
- [X] T007 [P] Create template configuration JSON for contractor_1 in data/contractor_1/config.json
- [X] T008 [P] Create template configuration JSON for contractor_2 in data/contractor_2/config.json
- [X] T009 Implement template loader in vlm_pdf_recognizer/templates/template_loader.py (load template images and JSON configs)
- [X] T010 Implement feature cache module in vlm_pdf_recognizer/templates/template_cache.py (save/load SIFT features via pickle)
- [X] T011 Create GoldenTemplate data class in vlm_pdf_recognizer/templates/__init__.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Process Single Scanned Document (Priority: P1) 🎯 MVP

**Goal**: Process single scanned document (PDF or image), classify template, remove watermarks, align geometrically, extract and visualize ROIs

**Independent Test**: Provide one scanned document, verify (1) correct template identified, (2) document aligned, (3) ROI bounding boxes correctly overlaid on output image

### Implementation for User Story 1

#### PDF Conversion Module

- [X] T012 [US1] Implement PDF to image conversion in vlm_pdf_recognizer/preprocessing/pdf_converter.py (PyMuPDF integration, preserve dimensions)
- [X] T013 [US1] Add error handling for corrupted PDFs and invalid files in pdf_converter.py

#### Watermark Removal Module

- [X] T014 [US1] Implement HSV thresholding watermark removal in vlm_pdf_recognizer/preprocessing/watermark_removal.py (V < 180 for dark content, S > 100 for red stamps)
- [X] T015 [US1] Add morphological operations for noise cleanup in watermark_removal.py
- [X] T016 [US1] Implement binarization (output 0 or 255 only) in watermark_removal.py

#### Feature Extraction Module

- [X] T017 [US1] Implement SIFT feature extraction in vlm_pdf_recognizer/alignment/feature_extractor.py (cv2.SIFT_create with optimized parameters)
- [X] T018 [US1] Add feature extraction for both template and input documents in feature_extractor.py

#### Template Matching Module

- [X] T019 [US1] Implement FLANN-based feature matching in vlm_pdf_recognizer/alignment/template_matcher.py (Lowe's ratio test = 0.7)
- [X] T020 [US1] Implement voting mechanism for template selection in template_matcher.py (max inlier count wins)
- [X] T021 [US1] Add UnknownDocumentError exception for inlier_count < 50 in template_matcher.py

#### Geometric Correction Module

- [X] T022 [US1] Implement homography computation with RANSAC in vlm_pdf_recognizer/alignment/geometric_corrector.py (5.0 pixel threshold)
- [X] T023 [US1] Implement perspective warp transformation in geometric_corrector.py (cv2.warpPerspective)
- [X] T024 [US1] Add AlignmentError exception handling in geometric_corrector.py

#### ROI Extraction Module

- [X] T025 [US1] Implement ROI extraction from aligned image in vlm_pdf_recognizer/extraction/roi_extractor.py (crop based on JSON coordinates)
- [X] T026 [US1] Implement bounding box overlay visualization in roi_extractor.py (green boxes, ROI labels)
- [X] T027 [US1] Add ROIExtractionError for out-of-bounds coordinates in roi_extractor.py

#### Pipeline Orchestration

- [X] T028 [US1] Implement main processing pipeline in vlm_pdf_recognizer/pipeline.py (orchestrate preprocessing → matching → alignment → extraction)
- [X] T029 [US1] Add ProcessingResult data structure in vlm_pdf_recognizer/pipeline.py (template_id, confidence, output_path, timing)
- [X] T030 [US1] Implement output image saving to output/ directory with metadata JSON in pipeline.py
- [X] T031 [US1] Add processing time tracking in pipeline.py

#### Command-Line Interface

- [X] T032 [US1] Implement CLI for single document processing in vlm_pdf_recognizer/cli.py (argparse, --output flag)
- [X] T033 [US1] Add GPU detection and fallback logic in cli.py (check cv2.cuda availability)
- [X] T034 [US1] Add verbose logging option in cli.py

**Checkpoint**: At this point, User Story 1 should be fully functional - single document can be processed end-to-end

---

## Phase 4: User Story 2 - Handle Unknown or Poor Quality Documents (Priority: P2)

**Goal**: Gracefully handle documents that don't match templates or have insufficient quality, returning clear error messages

**Independent Test**: Provide (1) non-matching document type, (2) blurred document, (3) document with <50 matches, verify appropriate error messages

### Implementation for User Story 2

- [ ] T035 [US2] Enhance error messages in template_matcher.py (include match counts, template scores)
- [ ] T036 [US2] Add confidence score logging in pipeline.py (log warning if confidence < 0.5)
- [ ] T037 [US2] Implement detailed error reporting in cli.py (show which templates were tried, match counts)
- [ ] T038 [US2] Add document quality validation in preprocessing module (check resolution, file integrity)
- [ ] T039 [US2] Create error summary output in cli.py (save error details to errors.log)

**Checkpoint**: Error handling complete - system provides actionable feedback for failures

---

## Phase 5: User Story 3 - Batch Process Multiple Documents (Priority: P3)

**Goal**: Process entire directory of mixed document types, skip failures, continue processing, generate summary report

**Independent Test**: Process folder with 10 mixed documents (3 enterprise_1, 4 contractor_1, 3 contractor_2), verify each classified correctly and outputs organized by template type

### Implementation for User Story 3

- [ ] T040 [US3] Implement batch processor in vlm_pdf_recognizer/batch_processor.py (iterate through directory, skip failures per FR-024)
- [ ] T041 [US3] Add BatchProcessingJob data structure in batch_processor.py (track completed_count, failed_count, file list)
- [ ] T042 [US3] Implement error logging for failed documents in batch_processor.py (capture file_path, error_message)
- [ ] T043 [US3] Add progress indicator callback support in batch_processor.py (optional progress updates)
- [ ] T044 [US3] Implement summary report generation in batch_processor.py (success/failure counts per template)
- [ ] T045 [US3] Organize outputs by template type in batch_processor.py (output/{template_id}/ subdirectories)
- [ ] T046 [US3] Add batch command to CLI in cli.py (batch subcommand with input_dir, output_dir)
- [ ] T047 [US3] Implement failed documents error log in batch_processor.py (output/errors.log with details)

**Checkpoint**: All user stories complete - system can process single documents, handle errors, and batch process

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T048 [P] Add docstrings to all public functions and classes across all modules
- [ ] T049 [P] Create quickstart example scripts in examples/ (example_single.py, example_batch.py)
- [ ] T050 [P] Validate against quickstart.md scenarios (test installation, setup, basic usage)
- [ ] T051 [P] Add performance logging for each pipeline stage in pipeline.py (watermark removal time, matching time, alignment time)
- [ ] T052 [P] Implement feature cache warming on startup in template_loader.py (pre-load all template features)
- [ ] T053 Code cleanup and remove debug print statements across all modules
- [ ] T054 Add error recovery hints in error messages (suggest resolution >= 150 DPI, check watermark colors)
- [ ] T055 Create data/README.md with template setup instructions and JSON schema documentation
- [ ] T056 Test CPU vs GPU result consistency (verify same homography matrix, alignment output)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion (T001-T005) - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (T006-T011) completion
- **User Story 2 (Phase 4)**: Depends on User Story 1 completion (needs error handling context)
- **User Story 3 (Phase 5)**: Depends on User Story 1 completion (batch uses single document pipeline)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Independent - can start after Foundational phase
- **User Story 2 (P2)**: Depends on US1 (enhances error handling of existing pipeline)
- **User Story 3 (P3)**: Depends on US1 (delegates to single document pipeline)

### Within Each User Story

**User Story 1 task dependencies**:
1. T012-T013 (PDF conversion) - independent, can run early
2. T014-T016 (Watermark removal) - can run parallel with PDF conversion
3. T017-T018 (Feature extraction) - needs template_loader (T009)
4. T019-T021 (Template matching) - needs feature_extractor (T017)
5. T022-T024 (Geometric correction) - needs template_matcher (T019)
6. T025-T027 (ROI extraction) - needs geometric_corrector (T022)
7. T028-T031 (Pipeline) - needs all above modules
8. T032-T034 (CLI) - needs pipeline (T028)

### Parallel Opportunities

- **Phase 1**: T003, T004, T005 can run in parallel
- **Phase 2**: T007, T008 can run in parallel (after T006 completes)
- **Phase 3 (US1)**:
  - T012-T013 (PDF) can run parallel with T014-T016 (watermark)
  - All modules within a layer can be developed in parallel once prerequisites met
- **Phase 6**: T048, T049, T050, T051, T052 can all run in parallel

---

## Parallel Example: User Story 1

```bash
# Early parallel development (after Foundational complete):
Task T012: "Implement PDF to image conversion in vlm_pdf_recognizer/preprocessing/pdf_converter.py"
Task T014: "Implement HSV thresholding watermark removal in vlm_pdf_recognizer/preprocessing/watermark_removal.py"

# After feature extraction ready:
Task T019: "Implement FLANN-based feature matching in vlm_pdf_recognizer/alignment/template_matcher.py"
Task T017: "Implement SIFT feature extraction (can develop alongside matcher using mocks)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T011) - CRITICAL
3. Complete Phase 3: User Story 1 (T012-T034)
4. **STOP and VALIDATE**: Test with real scanned documents
5. Verify alignment quality, ROI extraction accuracy, performance (<10s CPU)
6. Demo/validate before adding more features

### Incremental Delivery

1. Setup + Foundational → Can load templates, ready for processing
2. Add User Story 1 → Test with single documents → MVP functional ✅
3. Add User Story 2 → Better error messages → More robust system
4. Add User Story 3 → Batch capability → Production-ready for volume
5. Polish → Performance optimization, documentation

### Sequential Execution (Single Developer)

Recommended order for solo development:

1. Phase 1 (Setup): T001 → T002 → T003, T004, T005 in any order
2. Phase 2 (Foundational): T006 → T007, T008 (parallel) → T009 → T010 → T011
3. Phase 3 (US1): Follow dependency order
   - T012, T014 (parallel)
   - T013, T015, T016
   - T017, T018
   - T019, T020, T021
   - T022, T023, T024
   - T025, T026, T027
   - T028, T029, T030, T031
   - T032, T033, T034
4. Phase 4 (US2): T035 → T036 → T037 → T038 → T039
5. Phase 5 (US3): T040 → T041 → T042 → T043 → T044 → T045 → T046 → T047
6. Phase 6 (Polish): Any order, all tasks are parallel

---

## Notes

- No test tasks included - specification doesn't request TDD approach
- Focus on implementation with validation via quickstart.md scenarios
- Each module should be testable manually via Python REPL or simple scripts
- Commit after completing each phase or major module
- Use research.md recommendations for parameter values (HSV threshold=180, RANSAC threshold=5.0, min matches=50)
- Refer to data-model.md for entity structures and validation rules
- GPU acceleration is auto-detected, no separate code path needed (OpenCV handles fallback)
- Total tasks: 56
  - Setup: 5 tasks
  - Foundational: 6 tasks
  - User Story 1: 23 tasks (MVP)
  - User Story 2: 5 tasks
  - User Story 3: 8 tasks
  - Polish: 9 tasks
