# Tasks: VLM Auxiliary ROI Comparison System

**Input**: Design documents from `/specs/003-vlm-auxiliary-roi-comparison/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Tests are included in this plan following existing project pytest infrastructure.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

This project uses single project structure:
- **Package**: `vlm_pdf_recognizer/` at repository root
- **Tests**: `tests/unit/` and `tests/integration/`
- **Config**: `update_configs.py`, `main.py` at root
- **Data**: `data/{template_id}/` for template features

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and verification of existing structure

- [ ] T001 Verify existing project structure matches plan.md (vlm_pdf_recognizer/ package with alignment/, recognition/, extraction/ modules)
- [ ] T002 Verify pytest and existing test infrastructure is functional (run `pytest tests/`)
- [ ] T003 [P] Create directory structure for new modules: vlm_pdf_recognizer/alignment/roi_comparator.py and blank_roi_cache.py placeholders

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Blank template ROI feature extraction that MUST be complete before ANY user story can use auxiliary comparison

**⚠️ CRITICAL**: No user story work can use auxiliary comparison until blank features are generated

### Blank ROI Feature Extraction (Configuration)

- [ ] T004 Implement extract_blank_roi_features() function in update_configs.py to extract SIFT features from blank template ROIs
- [ ] T005 Modify update_config_from_labelme() in update_configs.py to load template image and call extract_blank_roi_features()
- [ ] T006 Add blank ROI feature saving logic to update_configs.py using numpy.savez() to create data/{template_id}/blank_roi_features.npz
- [ ] T007 Run update_configs.py to generate blank ROI features for all 3 templates (contractor_1, contractor_2, enterprise_1)
- [ ] T008 Verify blank ROI features generated correctly: check data/{template_id}/blank_roi_features.npz exists and data/{template_id}/rois/blank_{field_id}.png images exist

### Blank ROI Feature Cache (Runtime Loading)

- [ ] T009 [P] Create BlankROIFeatures dataclass in vlm_pdf_recognizer/alignment/blank_roi_cache.py with validation
- [ ] T010 [P] Implement BlankROIFeatureCache class in vlm_pdf_recognizer/alignment/blank_roi_cache.py with load_from_directory(), get_features(), and has_features() methods
- [ ] T011 Integrate BlankROIFeatureCache into DocumentProcessor.__init__() in vlm_pdf_recognizer/pipeline.py to load and cache blank features at startup
- [ ] T012 Add logging to DocumentProcessor initialization for blank feature loading status (loaded templates count, failed templates warning)

**Checkpoint**: Foundation ready - blank features generated and cached, user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Blank Template ROI Feature Extraction (Priority: P1) 🎯 MVP

**Goal**: Enable configuration script to extract and store blank template ROI features for baseline comparison

**Independent Test**: Run update_configs.py, verify blank ROI features are saved to data/{template_id}/blank_roi_features.npz and blank ROI images to data/{template_id}/rois/, validate feature count matches template field count

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T013 [P] [US1] Unit test for blank feature extraction in tests/unit/test_blank_roi_extraction.py - test valid template image extracts features correctly
- [ ] T014 [P] [US1] Unit test for missing template handling in tests/unit/test_blank_roi_extraction.py - test graceful error when template image missing
- [ ] T015 [P] [US1] Unit test for invalid ROI coordinates in tests/unit/test_blank_roi_extraction.py - test handling of out-of-bounds coordinates
- [ ] T016 [P] [US1] Unit test for blank feature cache loading in tests/unit/test_blank_roi_cache.py - test load_from_directory() with valid .npz files
- [ ] T017 [P] [US1] Unit test for missing .npz file handling in tests/unit/test_blank_roi_cache.py - test template added to failed_templates
- [ ] T018 [P] [US1] Unit test for get_features() in tests/unit/test_blank_roi_cache.py - test retrieval of existing and missing features

### Implementation for User Story 1

(Already completed in Phase 2 Foundational - no additional implementation needed)

**Checkpoint**: Blank template features are extracted, saved, and can be loaded. US1 is fully functional and testable independently.

---

## Phase 4: User Story 2 - Auxiliary ROI Comparison Before VLM Inference (Priority: P1) 🎯 MVP

**Goal**: Compare document ROIs against blank template ROIs to determine if field is filled before running expensive VLM inference

**Independent Test**: Process a test document with mixed filled/unfilled fields, verify auxiliary_has_content column shows correct True/False values based on ROI similarity, confirm VLM is skipped for empty fields (vlm_has_content=None)

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T019 [P] [US2] Unit test for compare_roi_to_blank() in tests/unit/test_roi_comparator.py - test identical ROIs return similarity ~1.0, auxiliary_has_content=False
- [ ] T020 [P] [US2] Unit test for filled vs blank comparison in tests/unit/test_roi_comparator.py - test filled ROI vs blank returns similarity <0.6, auxiliary_has_content=True
- [ ] T021 [P] [US2] Unit test for empty vs blank comparison in tests/unit/test_roi_comparator.py - test empty ROI vs blank returns similarity >=0.6, auxiliary_has_content=False
- [ ] T022 [P] [US2] Unit test for insufficient features in tests/unit/test_roi_comparator.py - test ROI with <4 keypoints returns error_message, auxiliary_has_content=None
- [ ] T023 [P] [US2] Unit test for threshold boundary in tests/unit/test_roi_comparator.py - test similarity_score exactly 0.6 returns auxiliary_has_content=False
- [ ] T024 [P] [US2] Integration test for VLM skip on empty fields in tests/integration/test_vlm_pipeline.py - test empty fields have auxiliary_has_content=False and vlm_has_content=None

### Implementation for User Story 2

- [ ] T025 [P] [US2] Create ROIComparisonResult dataclass in vlm_pdf_recognizer/alignment/roi_comparator.py with all required fields
- [ ] T026 [US2] Implement compare_roi_to_blank() function in vlm_pdf_recognizer/alignment/roi_comparator.py using existing feature_extractor and template_matcher functions
- [ ] T027 [US2] Add error handling to compare_roi_to_blank() for insufficient features, matching failures, and RANSAC failures with appropriate fallback logic
- [ ] T028 [US2] Extend RecognitionResult dataclass in vlm_pdf_recognizer/recognition/vlm_recognizer.py to add auxiliary_has_content, auxiliary_similarity_score, auxiliary_comparison_time_ms fields
- [ ] T029 [US2] Modify VLMRecognizer._recognize_field() in vlm_pdf_recognizer/recognition/vlm_recognizer.py to add auxiliary comparison step before VLM inference for non-title, non-checkbox fields
- [ ] T030 [US2] Implement conditional VLM execution in VLMRecognizer._recognize_field() - skip VLM if auxiliary_has_content=False, run VLM if auxiliary_has_content=True or None
- [ ] T031 [US2] Update VLMRecognizer.process_document() signature in vlm_pdf_recognizer/recognition/vlm_recognizer.py to accept blank_roi_cache parameter and pass to _recognize_field()
- [ ] T032 [US2] Modify main.py around line 150 to pass processor.blank_roi_cache to vlm_recognizer.process_document()

**Checkpoint**: Auxiliary comparison is functional, VLM is conditionally skipped based on similarity. US2 works independently.

---

## Phase 5: User Story 3 - Integrated Output with Auxiliary Comparison Results (Priority: P1) 🎯 MVP

**Goal**: Output both VLM and auxiliary comparison results for transparency and debugging

**Independent Test**: Process a test document, verify output JSON/CSV contains both vlm_has_content and auxiliary_has_content columns, verify values correctly populated from ROI comparison logic

### Tests for User Story 3

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T033 [P] [US3] Unit test for to_json_dict() extension in tests/unit/test_output.py - test JSON output contains auxiliary_has_content and auxiliary_similarity_score fields
- [ ] T034 [P] [US3] Integration test for output schema in tests/integration/test_vlm_pipeline.py - test processed document output has all expected auxiliary columns
- [ ] T035 [P] [US3] Integration test for title field handling in tests/integration/test_vlm_pipeline.py - test title fields have auxiliary_has_content=None

### Implementation for User Story 3

- [ ] T036 [P] [US3] Modify DocumentRecognitionOutput.to_json_dict() in vlm_pdf_recognizer/recognition/vlm_recognizer.py to add auxiliary_has_content and auxiliary_similarity_score to field output
- [ ] T037 [P] [US3] Update save_batch_summary_with_vlm() in vlm_pdf_recognizer/output.py to preserve auxiliary fields in JSON export (if needed - verify current implementation)

**Checkpoint**: Outputs include both auxiliary and VLM data. US3 works independently, allows downstream analysis of both methods.

---

## Phase 6: User Story 4 - Auxiliary-Priority Result Validation Logic (Priority: P1) 🎯 MVP

**Goal**: Calculate document validation status using auxiliary comparison results instead of VLM results for more accurate presence detection

**Independent Test**: Process documents with known completion status, verify results field correctly reflects validation using auxiliary_has_content values with correct logic (VX1 priority, date OR, other fields AND)

### Tests for User Story 4

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T038 [P] [US4] Unit test for calculate_results_status() with auxiliary in tests/unit/test_vlm_recognizer.py - test VX1=True returns results=False
- [ ] T039 [P] [US4] Unit test for date field OR logic in tests/unit/test_vlm_recognizer.py - test at least one date field with auxiliary_has_content=True
- [ ] T040 [P] [US4] Unit test for other fields AND logic in tests/unit/test_vlm_recognizer.py - test all non-date fields have auxiliary_has_content=True
- [ ] T041 [P] [US4] Unit test for auxiliary fallback to VLM in tests/unit/test_vlm_recognizer.py - test uses VLM has_content when auxiliary_has_content=None
- [ ] T042 [P] [US4] Integration test for validation accuracy in tests/integration/test_vlm_pipeline.py - test results field matches expected validation status on known documents

### Implementation for User Story 4

- [ ] T043 [US4] Update DocumentRecognitionOutput.calculate_results_status() in vlm_pdf_recognizer/recognition/vlm_recognizer.py to use auxiliary_has_content with fallback to VLM has_content
- [ ] T044 [US4] Implement VX1 priority check using existing heuristic has_content (not auxiliary) in calculate_results_status()
- [ ] T045 [US4] Implement date field OR logic using auxiliary_has_content for year, month, date fields in calculate_results_status()
- [ ] T046 [US4] Implement other fields AND logic using auxiliary_has_content for all non-title, non-date, non-VX1 fields in calculate_results_status()

**Checkpoint**: Validation logic uses auxiliary results, accuracy improved. US4 works independently with correct validation outcomes.

---

## Phase 7: User Story 5 - Color-Coded Visualization with Auxiliary Results (Priority: P2)

**Goal**: Generate visualization images with ROI boxes colored based on auxiliary comparison results for quick visual inspection

**Independent Test**: Process a document, verify visualization image shows green boxes for filled fields (auxiliary_has_content=True) and red boxes for unfilled fields (auxiliary_has_content=False)

### Tests for User Story 5

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T047 [P] [US5] Integration test for visualization coloring in tests/integration/test_vlm_pipeline.py - test visualization image uses auxiliary-based colors (verify by checking pixel colors in ROI boxes)
- [ ] T048 [P] [US5] Integration test for title field visualization in tests/integration/test_vlm_pipeline.py - test title fields use original template color

### Implementation for User Story 5

- [ ] T049 [US5] Modify save_vlm_visualization() in vlm_pdf_recognizer/output.py to use auxiliary_has_content for ROI box coloring instead of vlm_has_content
- [ ] T050 [US5] Update ROI box color logic in save_vlm_visualization() - green for auxiliary_has_content=True, red for auxiliary_has_content=False, original color for title fields
- [ ] T051 [US5] Update label format in save_vlm_visualization() to show auxiliary status: "{field_id}: {auxiliary_has_content}"

**Checkpoint**: Visualizations use auxiliary coloring, providing quick visual feedback. US5 enhances user experience independently.

---

## Phase 8: User Story 6 - Checkbox Recognition Preservation (Priority: P1) 🎯 MVP

**Goal**: Ensure VX1 and VX2 checkbox fields continue using existing heuristic method without auxiliary comparison

**Independent Test**: Process documents with checked/unchecked VX1 and VX2 boxes, verify recognition results match existing behavior (heuristic-primary), confirm auxiliary_has_content=None for checkboxes

### Tests for User Story 6

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T052 [P] [US6] Integration test for checkbox preservation in tests/integration/test_backward_compat.py - test VX1 and VX2 use existing heuristic, not auxiliary
- [ ] T053 [P] [US6] Unit test for checkbox auxiliary skip in tests/unit/test_vlm_recognizer.py - test checkbox fields have auxiliary_has_content=None

### Implementation for User Story 6

- [ ] T054 [US6] Verify VLMRecognizer._recognize_field() in vlm_pdf_recognizer/recognition/vlm_recognizer.py skips auxiliary for checkbox field types
- [ ] T055 [US6] Verify validation logic in calculate_results_status() uses VX1 heuristic has_content, not auxiliary_has_content

**Checkpoint**: Checkbox recognition unchanged, no regression. US6 preserves existing accuracy for checkboxes.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements affecting multiple user stories, backward compatibility, and documentation

- [ ] T056 [P] Integration test for backward compatibility with missing blank features in tests/integration/test_backward_compat.py - test system falls back to VLM-only when blank features unavailable
- [ ] T057 [P] Integration test for end-to-end auxiliary pipeline in tests/integration/test_vlm_pipeline.py - test full pipeline from config to output with auxiliary enabled
- [ ] T058 [P] Unit test for performance validation in tests/unit/test_roi_comparator.py - test auxiliary comparison completes in <100ms per ROI
- [ ] T059 [P] Unit test for memory footprint in tests/unit/test_blank_roi_cache.py - test cache memory usage <20MB for all templates
- [ ] T060 [P] Add comprehensive logging for auxiliary comparison metrics (similarity scores, inlier counts, decisions) in vlm_pdf_recognizer/alignment/roi_comparator.py
- [ ] T061 [P] Add warning logs for missing blank features in BlankROIFeatureCache.load_from_directory() in vlm_pdf_recognizer/alignment/blank_roi_cache.py
- [ ] T062 [P] Update CLAUDE.md with auxiliary comparison feature summary and key files changed
- [ ] T063 Code review and refactoring for consistency with existing codebase style
- [ ] T064 Run full test suite with coverage analysis: `pytest tests/ -v --cov=vlm_pdf_recognizer`
- [ ] T065 Validate against quickstart.md checklist (all items marked complete)
- [ ] T066 Performance testing with real document batch (100+ documents) to validate <50ms per ROI and overall throughput improvement

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-8)**: All depend on Foundational phase completion
  - User stories can proceed in parallel after Phase 2 (if staffed)
  - Or sequentially in priority order for MVP approach
- **Polish (Phase 9)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories (actually completed in Phase 2)
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Depends on US1 for blank features but otherwise independent
- **User Story 3 (P1)**: Can start after US2 completion - Needs auxiliary comparison working to output results
- **User Story 4 (P1)**: Can start after US2 completion - Needs auxiliary comparison working for validation
- **User Story 5 (P2)**: Can start after US2 completion - Needs auxiliary comparison working for visualization
- **User Story 6 (P1)**: Can start after US2 completion - Verifies checkbox bypass logic is working

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Tests can run in parallel (all marked [P])
- Implementation tasks follow dependencies noted in descriptions
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- All tests for a user story can run in parallel
- Implementation tasks within a story marked [P] can run in parallel
- User Stories 3, 4, 5, 6 can be worked on in parallel after US2 completes (if team capacity allows)

---

## Parallel Example: User Story 2

```bash
# Launch all tests for User Story 2 together:
Task: "Unit test for compare_roi_to_blank() in tests/unit/test_roi_comparator.py - test identical ROIs"
Task: "Unit test for filled vs blank comparison in tests/unit/test_roi_comparator.py"
Task: "Unit test for empty vs blank comparison in tests/unit/test_roi_comparator.py"
Task: "Unit test for insufficient features in tests/unit/test_roi_comparator.py"
Task: "Unit test for threshold boundary in tests/unit/test_roi_comparator.py"
Task: "Integration test for VLM skip on empty fields in tests/integration/test_vlm_pipeline.py"

# Launch parallelizable implementation tasks together:
Task: "Create ROIComparisonResult dataclass in vlm_pdf_recognizer/alignment/roi_comparator.py"
# (This is the only [P] task in US2 implementation)
```

---

## Implementation Strategy

### MVP First (User Stories 1, 2, 4, 6 Only - Core Functionality)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - generates blank features, blocks all stories)
3. Complete Phase 3: User Story 1 (configuration feature extraction - included in Phase 2)
4. Complete Phase 4: User Story 2 (auxiliary comparison and conditional VLM)
5. Complete Phase 6: User Story 4 (auxiliary-priority validation)
6. Complete Phase 8: User Story 6 (checkbox preservation)
7. **STOP and VALIDATE**: Test core functionality independently
8. Deploy/demo if ready

**MVP Deliverable**: Auxiliary comparison working, validation logic using auxiliary results, VLM conditionally skipped, checkboxes preserved. This is the minimum viable feature.

### Full Feature (Add Output and Visualization)

9. Complete Phase 5: User Story 3 (integrated output with auxiliary columns)
10. Complete Phase 7: User Story 5 (color-coded visualization)
11. Complete Phase 9: Polish (tests, docs, performance validation)
12. **VALIDATE**: Full test suite, quickstart checklist, performance benchmarks
13. Deploy complete feature

### Incremental Delivery

1. Complete Setup + Foundational → Blank features generated
2. Add User Stories 1+2 → Test auxiliary comparison works → Deploy/Demo (MVP partial)
3. Add User Stories 4+6 → Test validation logic → Deploy/Demo (MVP complete!)
4. Add User Story 3 → Test outputs → Deploy/Demo (transparency added)
5. Add User Story 5 → Test visualization → Deploy/Demo (full feature)
6. Polish → Final validation → Production ready

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (critical path)
2. Once Foundational is done and US2 is complete:
   - Developer A: User Story 3 (output schema)
   - Developer B: User Story 4 (validation logic)
   - Developer C: User Story 5 (visualization)
   - Developer D: User Story 6 (checkbox preservation) + Phase 9 tests
3. Stories integrate independently, converge for final testing

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Tests follow existing pytest infrastructure (unit/, integration/)
- Reuse existing SIFT infrastructure from alignment module
- Maintain backward compatibility - missing blank features → VLM-only fallback
