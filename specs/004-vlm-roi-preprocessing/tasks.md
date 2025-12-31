# Tasks: VLM-Assisted ROI Content Detection with Image Preprocessing

**Input**: Design documents from `/specs/004-vlm-roi-preprocessing/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Not explicitly requested in specification - focusing on implementation and manual validation

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Python project at repository root: `vlm_pdf_recognizer/`, `tests/`, `update_configs.py`, `main.py`
- Configuration: `vlm_pdf_recognizer/recognition/config.py`
- Data storage: `data/{template_id}/blank_rois/`
- Output: `output/processed_rois/`

---

## Phase 1: Setup (Configuration Infrastructure)

**Purpose**: Add preprocessing configuration parameters to enable the 6-step pipeline

- [X] T001 Add preprocessing configuration parameters to vlm_pdf_recognizer/recognition/config.py (DIFFERENCE_THRESHOLD=25, SATURATION_BOOST_FACTOR=2.0, MORPHOLOGY_LINE_RATIO=0.3, MIN_BLOB_AREA=20, COMPONENT_COUNT_THRESHOLD=3, INK_RATIO_THRESHOLD=0.005, DEBUG_SAVE_INTERMEDIATE_IMAGES=False, PNG_COMPRESSION_LEVEL=6, PRE_ALLOCATE_WORK_BUFFERS=True, MAX_ROI_DIMENSION=1000)
- [X] T002 Verify config.py loads without errors by importing vlm_pdf_recognizer.recognition.config module

**Checkpoint**: Configuration parameters ready for use by preprocessing pipeline

---

## Phase 2: Foundational (Blank Template ROI Generation)

**Purpose**: Extend update_configs.py to generate blank template ROI images that serve as reference baselines

**⚠️ CRITICAL**: This phase MUST be complete before ANY preprocessing pipeline work can begin

- [X] T003 Add extract_and_save_blank_rois function to update_configs.py to extract blank template ROIs using existing roi_extractor.extract_rois logic
- [X] T004 Integrate extract_and_save_blank_rois into update_configs.py main() function after existing SIFT feature extraction
- [X] T005 Modify extract_and_save_blank_rois to save each blank ROI as PNG to data/{template_id}/blank_rois/{field_id}.png
- [X] T006 Run update_configs.py and verify blank_rois directories created with PNG files for all three templates (contractor_1, contractor_2, enterprise_1)
- [X] T007 Visually inspect sample blank ROI images (signature.png, stamp1.png) to confirm they show blank template regions

**Checkpoint**: Foundation ready - blank template ROIs generated and verified for all templates

---

## Phase 3: User Story 1 - Template ROI Baseline Generation (Priority: P1) 🎯 MVP

**Goal**: Generate blank template ROI images during system initialization to serve as reference baselines for preprocessing

**Independent Test**: Run update_configs.py with template images, verify that blank template ROI images are saved to data/{template_id}/blank_rois/{field_id}.png for each field, and can be loaded for comparison

### Implementation for User Story 1

- [X] T008 [US1] Create BlankTemplateROICache class in vlm_pdf_recognizer/alignment/blank_template_roi_cache.py with load_blank_rois, get_blank_roi, and get_loaded_count methods
- [X] T009 [US1] Implement load_blank_rois method to read PNG files from data/{template_id}/blank_rois/ directory and cache in memory dict
- [X] T010 [US1] Implement get_blank_roi method to retrieve specific blank ROI by template_id and field_id from cache
- [X] T011 [US1] Implement get_loaded_count method to return total number of loaded blank ROIs across all templates
- [X] T012 [US1] Add error handling to BlankTemplateROICache for missing blank_rois directory (log warning, return empty dict)
- [X] T013 [US1] Integrate BlankTemplateROICache into vlm_pdf_recognizer/pipeline.py DocumentProcessor.__init__ method
- [X] T014 [US1] Pre-load all blank ROIs in DocumentProcessor.__init__ by calling blank_roi_cache.load_blank_rois for each template_id
- [X] T015 [US1] Update main.py to display blank ROI cache status (loaded count) during VLM initialization section
- [X] T016 [US1] Test BlankTemplateROICache by running python main.py and verifying "Loaded blank features: N templates" message appears
- [X] T017 [US1] Verify BlankTemplateROICache.get_blank_roi returns valid np.ndarray for known field_id (signature, stamp1) and None for unknown field_id

**Checkpoint**: Blank template ROI cache fully functional and integrated into main.py execution flow

---

## Phase 4: User Story 2 - ROI Content Detection via Template Difference Preprocessing (Priority: P1)

**Goal**: Implement 6-step preprocessing pipeline to detect user content in ROI fields using template difference and morphological operations

**Independent Test**: Process a test document with mixed filled/unfilled fields, verify that preprocessing pipeline correctly identifies filled fields (has_content=True) and empty fields (has_content=False), with intermediate processing images saved to output/processed_rois/ for inspection

### Implementation for User Story 2

- [X] T018 [US2] Create PreprocessingResult dataclass in vlm_pdf_recognizer/recognition/roi_preprocessor.py with fields: field_id, has_content, ink_ratio, component_count, processing_time_ms, error_message
- [X] T019 [US2] Create ROIPreprocessor class in vlm_pdf_recognizer/recognition/roi_preprocessor.py with __init__(save_debug_images, output_dir) constructor
- [X] T020 [US2] Implement ROIPreprocessor.__init__ to pre-allocate work buffers if config.PRE_ALLOCATE_WORK_BUFFERS=True and cache morphological kernels (horizontal_kernels, vertical_kernels, noise_kernel)
- [X] T021 [US2] Implement ROIPreprocessor.preprocess_roi method signature accepting (doc_roi_image, blank_template_roi, field_id, document_name) and returning PreprocessingResult
- [X] T022 [US2] Implement Step 1: _extract_saturation_channel method to convert BGR to HSV, extract saturation channel, apply SATURATION_BOOST_FACTOR, blend with inverted value channel using SATURATION_VALUE_BLEND_RATIO
- [X] T023 [US2] Implement Step 2: _compute_template_difference method to convert saturation images to grayscale (if needed), compute absolute difference, apply binary threshold at DIFFERENCE_THRESHOLD
- [X] T024 [US2] Add CLAHE normalization to _compute_template_difference if config.USE_CLAHE_NORMALIZATION=True (clipLimit=2.0, tileGridSize=8x8)
- [X] T025 [US2] Implement Step 3: _remove_form_lines method to create horizontal kernel (width=ROI_width*MORPHOLOGY_LINE_RATIO, height=3) and vertical kernel (width=3, height=ROI_height*MORPHOLOGY_LINE_RATIO), apply morphological opening for each
- [X] T026 [US2] Cache morphological kernels in _remove_form_lines by ROI dimensions to avoid recomputation (store in self.horizontal_kernels and self.vertical_kernels dicts)
- [X] T027 [US2] Implement Step 4: _remove_noise method to apply morphological opening with 3x3 kernel, then use cv2.connectedComponentsWithStats to filter components with area < MIN_BLOB_AREA
- [X] T028 [US2] Implement Step 5: _extract_features method to count valid connected components (area >= MIN_BLOB_AREA, aspect_ratio <= MAX_ASPECT_RATIO=10, density >= MIN_COMPONENT_DENSITY=0.3), calculate ink_ratio as total_valid_area / total_pixels
- [X] T029 [US2] Implement Step 6: _determine_has_content method to return (True, reasoning) if (ink_ratio > INK_RATIO_THRESHOLD AND component_count >= COMPONENT_COUNT_THRESHOLD), else (False, reasoning)
- [X] T030 [US2] Integrate all 6 steps into preprocess_roi method with error handling (catch exceptions, return PreprocessingResult with has_content=None and error_message populated)
- [X] T031 [US2] Add ROI dimension mismatch handling in preprocess_roi: resize doc_roi_image to match blank_template_roi.shape if different, log warning if size difference >10%
- [X] T032 [US2] Implement _save_intermediate_images method to create output_dir/{document_name}/{field_id}/ directory and save 6 PNG files (01_saturation.png, 02_difference.png, 03_hline_removed.png, 04_vline_removed.png, 05_noise_removed.png, 06_final_binary.png) with config.PNG_COMPRESSION_LEVEL
- [X] T033 [US2] Implement _save_metadata method to write output_dir/{document_name}/{field_id}/metadata.json with preprocessing metrics (field_id, has_content, ink_ratio, component_count, processing_time_ms, thresholds_used, decision_reasoning, error_message)
- [X] T034 [US2] Call _save_intermediate_images and _save_metadata in preprocess_roi if self.save_debug_images=True and document_name is provided
- [X] T035 [US2] Add timing measurement in preprocess_roi using time.time() to populate PreprocessingResult.processing_time_ms
- [X] T036 [US2] Test ROIPreprocessor.preprocess_roi with blank ROI (white image) and verify has_content=False, ink_ratio<0.005, component_count<3
- [X] T037 [US2] Test ROIPreprocessor.preprocess_roi with signature ROI (black rectangle added) and verify has_content=True, ink_ratio>0.005, component_count>=1
- [X] T038 [US2] Enable DEBUG_SAVE_INTERMEDIATE_IMAGES=True and test preprocess_roi to verify 6 PNG files and metadata.json are saved to output/processed_rois/test_document/test_field/

**Checkpoint**: ROI Preprocessor 6-step pipeline fully implemented and independently tested with blank/filled ROI samples

---

## Phase 5: User Story 3 - Integrated VLM Recognition with Preprocessed Content Flags (Priority: P1)

**Goal**: Integrate preprocessing pipeline into VLM recognition workflow so preprocessing runs before VLM inference for all non-title, non-checkbox fields

**Independent Test**: Process a document where preprocessing detects content in 10 fields, verify that VLM runs on all fields, extracts text where present, and output contains both VLM_has_content and preprocessing_has_content columns with potentially different values for debugging

### Implementation for User Story 3

- [X] T039 [US3] Add preprocessing fields to RecognitionResult dataclass in vlm_pdf_recognizer/recognition/vlm_recognizer.py: preprocessing_has_content (bool|None), preprocessing_ink_ratio (float|None), preprocessing_component_count (int|None), preprocessing_time_ms (float|None)
- [X] T040 [US3] Modify VLMRecognizer._recognize_field method signature to accept blank_roi_cache parameter (BlankTemplateROICache instance)
- [X] T041 [US3] In VLMRecognizer._recognize_field for title fields: set preprocessing_has_content=None, preprocessing_ink_ratio=None, preprocessing_component_count=None, preprocessing_time_ms=None (skip preprocessing)
- [X] T042 [US3] In VLMRecognizer._recognize_field for checkbox fields: set preprocessing_has_content=None, preprocessing_ink_ratio=None, preprocessing_component_count=None, preprocessing_time_ms=None (use existing heuristic, skip preprocessing)
- [X] T043 [US3] In VLMRecognizer._recognize_field for other fields (text, number, stamp): add preprocessing step before VLM inference
- [X] T044 [US3] Implement preprocessing integration: if blank_roi_cache and template_id, call blank_roi_cache.get_blank_roi(template_id, field_id) to get blank_roi
- [X] T045 [US3] If blank_roi is not None, create ROIPreprocessor instance with save_debug_images from config.DEBUG_SAVE_INTERMEDIATE_IMAGES and output_dir="output/processed_rois"
- [X] T046 [US3] Call preprocessor.preprocess_roi(roi_image, blank_roi, field_id, document_name) and capture PreprocessingResult
- [X] T047 [US3] Populate RecognitionResult preprocessing fields from PreprocessingResult: preprocessing_has_content=result.has_content, preprocessing_ink_ratio=result.ink_ratio, preprocessing_component_count=result.component_count, preprocessing_time_ms=result.processing_time_ms
- [X] T048 [US3] Ensure VLM inference always runs for all non-title fields regardless of preprocessing_has_content value (for content extraction and debugging)
- [X] T049 [US3] Update VLMRecognizer.process_document method to pass blank_roi_cache to _recognize_field calls
- [X] T050 [US3] Modify main.py to pass processor.blank_roi_cache to vlm_recognizer.process_document when calling VLM recognition
- [X] T051 [US3] Test integration by running python main.py with a test document and verifying preprocessing_has_content appears in RecognitionResult for non-title, non-checkbox fields
- [X] T052 [US3] Verify preprocessing_has_content=None for title and checkbox fields, and bool (True/False) for other fields

**Checkpoint**: Preprocessing integrated into VLM recognition workflow, runs automatically before VLM inference for applicable fields

---

## Phase 6: User Story 4 - Auxiliary-Priority Validation Logic (Priority: P1)

**Goal**: Update document validation logic to use preprocessing_has_content as primary input instead of auxiliary_has_content or VLM has_content

**Independent Test**: Process documents with known completion status, verify that results field correctly reflects validation using preprocessing has_content values, matching existing logic (VX1=True means results=False, all date fields False means results=False, any other field False means results=False)

### Implementation for User Story 4

- [X] T053 [US4] Modify DocumentRecognitionOutput.calculate_results_status method in vlm_pdf_recognizer/recognition/vlm_recognizer.py to use preprocessing_has_content as primary input
- [X] T054 [US4] Update VX1 priority logic: if VX1.has_content is True (using heuristic result, NOT preprocessing), return False immediately
- [X] T055 [US4] Update date fields OR logic: check if any of (year, month, date) fields have (preprocessing_has_content if not None else has_content) == True, set date_valid accordingly
- [X] T056 [US4] Update other fields AND logic: for all non-title, non-date fields (excluding VX1), check if all have (preprocessing_has_content if not None else has_content) == True, set other_fields_valid accordingly
- [X] T057 [US4] Return final results = date_valid AND other_fields_valid from calculate_results_status
- [X] T058 [US4] Test calculate_results_status with mock RecognitionResults where VX1.preprocessing_has_content=True and verify results=False
- [X] T059 [US4] Test calculate_results_status with mock RecognitionResults where all date fields have preprocessing_has_content=False and verify results=False
- [X] T060 [US4] Test calculate_results_status with mock RecognitionResults where one date field has preprocessing_has_content=True, all other fields True, and verify results=True
- [X] T061 [US4] Test calculate_results_status with mock RecognitionResults where company1.preprocessing_has_content=False and verify results=False

**Checkpoint**: Validation logic updated to prioritize preprocessing results, maintains existing validation rules (VX1 priority, date OR, other AND)

---

## Phase 7: User Story 6 - Checkbox Recognition Preservation (Priority: P1)

**Goal**: Ensure VX1 and VX2 checkbox fields skip preprocessing and continue using existing heuristic pixel check method

**Independent Test**: Process documents with checked/unchecked VX1 and VX2 boxes, verify that recognition results match existing behavior (heuristic pixel counting), preprocessing pipeline is skipped for checkboxes, and validation logic uses heuristic results for VX1/VX2

### Implementation for User Story 6

- [X] T062 [US6] Verify VLMRecognizer._recognize_field skips preprocessing for field_schema.field_type == "checkbox" (already implemented in Phase 5 T042)
- [X] T063 [US6] Verify preprocessing_has_content=None for checkbox fields in RecognitionResult (already implemented in Phase 5 T042)
- [X] T064 [US6] Verify calculate_results_status uses VX1.has_content (heuristic result) not preprocessing_has_content for VX1 priority check (already implemented in Phase 6 T054)
- [X] T065 [US6] Test with document containing checked VX1 checkbox: verify heuristic detects checkmark, preprocessing_has_content=None, validation uses heuristic result
- [X] T066 [US6] Test with document containing unchecked VX2 checkbox: verify heuristic returns False, preprocessing_has_content=None, validation uses heuristic result

**Checkpoint**: Checkbox recognition unchanged, regression prevented for VX1/VX2 fields

---

## Phase 8: User Story 5 - Color-Coded Visualization with Preprocessing Results (Priority: P2)

**Goal**: Update visualization images to color ROI boxes based on preprocessing_has_content values (green=True, red=False)

**Independent Test**: Process a document and verify that visualization image shows green boxes around filled fields (preprocessing detected content) and red boxes around empty fields (preprocessing detected no content), with labels indicating preprocessing source

### Implementation for User Story 5

- [X] T067 [P] [US5] Add preprocessing output columns to DocumentRecognitionOutput.to_json_dict method in vlm_pdf_recognizer/recognition/vlm_recognizer.py: preprocessing_has_content, preprocessing_ink_ratio, preprocessing_component_count for each field
- [X] T068 [P] [US5] Update CSV export in vlm_pdf_recognizer/output.py to include preprocessing_{field_id}_has_content columns for each field
- [X] T069 [US5] Modify save_vlm_visualization function in vlm_pdf_recognizer/output.py to determine ROI box color based on preprocessing_has_content instead of auxiliary_has_content or VLM has_content
- [X] T070 [US5] Implement color logic in save_vlm_visualization: title fields use original color (green), checkbox fields use heuristic has_content (green=True, red=False), other fields use preprocessing_has_content (green=True, red=False, blue=None/error)
- [X] T071 [US5] Update ROI box label format in save_vlm_visualization to "{field_id}: {preprocessing_has_content}" or "{field_id}: {has_content}" for checkboxes
- [ ] T072 [US5] Test save_vlm_visualization with mock document containing 10 fields with preprocessing_has_content=True and 3 fields with preprocessing_has_content=False, verify 10 green boxes and 3 red boxes
- [ ] T073 [US5] Test save_vlm_visualization with field where preprocessing_has_content=None (error), verify blue box color
- [ ] T074 [US5] Run python main.py with test document and visually inspect output/*_visualization.png to confirm green/red color coding matches preprocessing results

**Checkpoint**: Visualization updated to show preprocessing results with color-coded ROI boxes

---

## Phase 9: User Story 7 - Processed ROI Image Export (Priority: P2)

**Goal**: Save intermediate preprocessing images for debugging and threshold tuning

**Independent Test**: Process a document with preprocessing enabled, verify that output/processed_rois/ contains subdirectories for each document with step-by-step images for each field

### Implementation for User Story 7

- [ ] T075 [US7] Enable debug image saving by setting environment variable: export DEBUG_ROI_PREPROCESSING=true before running main.py
- [ ] T076 [US7] Update vlm_pdf_recognizer/recognition/config.py to read DEBUG_SAVE_INTERMEDIATE_IMAGES from environment variable: os.getenv('DEBUG_ROI_PREPROCESSING', 'false').lower() == 'true'
- [ ] T077 [US7] Verify ROIPreprocessor._save_intermediate_images creates output/processed_rois/{document_name}/{field_id}/ directory structure (already implemented in Phase 4 T032)
- [ ] T078 [US7] Verify 6 intermediate images are saved: 01_saturation.png, 02_difference.png, 03_hline_removed.png, 04_vline_removed.png, 05_noise_removed.png, 06_final_binary.png (already implemented in Phase 4 T032)
- [ ] T079 [US7] Verify metadata.json is saved with preprocessing metrics (already implemented in Phase 4 T033)
- [ ] T080 [US7] Test with DEBUG_ROI_PREPROCESSING=true: run python main.py and verify output/processed_rois/{document_name}/ contains subdirectories for each field
- [ ] T081 [US7] Visually inspect intermediate images for a field with preprocessing_has_content=True: verify difference image shows content, morphology image shows line removal, final binary image shows detected blobs
- [ ] T082 [US7] Visually inspect intermediate images for a field with preprocessing_has_content=False: verify minimal difference, empty final binary image
- [ ] T083 [US7] Use metadata.json to understand decision reasoning: check if ink_ratio and component_count values match has_content determination

**Checkpoint**: Debug image export fully functional, enables visual debugging of preprocessing pipeline

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, cleanup, and documentation updates

- [ ] T084 [P] Run update_configs.py to regenerate blank template ROIs for all templates and verify no errors
- [ ] T085 [P] Run python main.py with multiple test documents (blank, partially filled, fully filled) and verify preprocessing correctly identifies filled/empty fields
- [ ] T086 [P] Compare preprocessing_has_content vs VLM_has_content vs AUX_has_content in VLM_results.json output and verify preprocessing provides more accurate has_content detection for stamp/signature fields
- [ ] T087 [P] Benchmark preprocessing performance: measure processing_time_ms per ROI and verify <100ms average (excluding debug image I/O)
- [ ] T088 [P] Test with DEBUG_ROI_PREPROCESSING=false and verify no debug images are saved, processing still works correctly
- [ ] T089 [P] Verify all existing tests still pass (zero regression): run pytest tests/ and check for any failures in VLM, checkbox, or template matching tests
- [ ] T090 [P] Update CLAUDE.md with preprocessing configuration parameters and usage instructions (already updated via update-agent-context.sh in Phase 1 of planning)
- [ ] T091 [P] Verify quickstart.md examples match actual implementation (review Phase 1-7 implementation tasks in quickstart.md)
- [ ] T092 Code cleanup: remove any debug print statements, ensure consistent logging levels (info, debug, warning, error)
- [ ] T093 [P] Threshold tuning: process real documents and adjust config.py thresholds (DIFFERENCE_THRESHOLD, INK_RATIO_THRESHOLD, etc.) based on accuracy results
- [ ] T094 Final validation: run complete pipeline (update_configs.py → python main.py) with production templates and verify results field accuracy matches or exceeds previous auxiliary comparison accuracy

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup (Phase 1) - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (Phase 4)**: Depends on Foundational (Phase 2) - No dependencies on other stories (but US2 uses blank ROIs from US1 at runtime)
- **User Story 3 (Phase 5)**: Depends on US1 (blank ROI cache) AND US2 (ROI preprocessor) - Integrates both
- **User Story 4 (Phase 6)**: Depends on US3 (preprocessing integration into VLM recognizer) - Uses preprocessing_has_content populated by US3
- **User Story 6 (Phase 7)**: Depends on US3 (VLM recognizer modifications) - Verifies checkbox skip logic
- **User Story 5 (Phase 8)**: Depends on US3 (RecognitionResult with preprocessing fields) - Exports preprocessing results
- **User Story 7 (Phase 9)**: Depends on US2 (ROI preprocessor with debug image saving) - Uses _save_intermediate_images
- **Polish (Phase 10)**: Depends on all user stories being complete

### User Story Dependencies

```
Phase 1 (Setup) → Phase 2 (Foundational)
                      ↓
    ┌─────────────────┴──────────────────────┐
    ↓                                        ↓
US1 (Blank ROI Cache)              US2 (ROI Preprocessor)
    ↓                                        ↓
    └──────────────┬─────────────────────────┘
                   ↓
          US3 (VLM Integration)
                   ↓
        ┌──────────┼──────────┐
        ↓          ↓          ↓
      US4        US6        US5
   (Validation) (Checkbox)  (Viz)

                 US7
           (Debug Export)
```

**Critical Path**: Phase 1 → Phase 2 → US1 → US2 → US3 → US4 (Core functionality)

**Parallel Opportunities**:
- Phase 1 T001-T002 can run together
- Phase 2 T003-T005 can run together (same file but different functions)
- US1 (Phase 3) and US2 (Phase 4) can run in parallel AFTER Phase 2 completes
- US5 (Phase 8 T067-T068) can run in parallel (different files)
- Phase 10 T084-T091 can run in parallel (different concerns)

### Within Each User Story

- US1: T008 → T009-T011 [P] → T012 → T013-T014 [P] → T015-T017
- US2: T018-T020 → T021 → T022-T029 (6-step implementation) → T030-T035 (integration) → T036-T038 (testing)
- US3: T039-T042 [P] (dataclass updates) → T043-T050 (VLM integration) → T051-T052 (testing)
- US4: T053-T057 (validation logic update) → T058-T061 (testing)
- US6: T062-T064 (verification) → T065-T066 (testing)
- US5: T067-T068 [P] (output updates) → T069-T074 (visualization updates)
- US7: T075-T079 (already implemented, verification) → T080-T083 (testing)

---

## Parallel Example: User Story 2 (ROI Preprocessor)

```bash
# After T021 (preprocess_roi signature), these can run in parallel:
Task: "Implement Step 1: _extract_saturation_channel" (T022)
Task: "Implement Step 2: _compute_template_difference" (T023)
Task: "Implement Step 3: _remove_form_lines" (T025)
Task: "Implement Step 4: _remove_noise" (T027)
Task: "Implement Step 5: _extract_features" (T028)
Task: "Implement Step 6: _determine_has_content" (T029)

# Then integrate all steps into preprocess_roi (T030)
```

---

## Implementation Strategy

### MVP First (User Stories 1, 2, 3, 4 - Core Preprocessing)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003-T007) - CRITICAL
3. Complete Phase 3: US1 Blank ROI Cache (T008-T017)
4. Complete Phase 4: US2 ROI Preprocessor (T018-T038)
5. Complete Phase 5: US3 VLM Integration (T039-T052)
6. Complete Phase 6: US4 Validation Logic (T053-T061)
7. **STOP and VALIDATE**: Test full pipeline with real documents, verify preprocessing correctly identifies filled/empty fields, results validation uses preprocessing_has_content
8. Deploy/demo if ready (MVP delivers core preprocessing functionality)

### Incremental Delivery

1. **Foundation**: Setup + Foundational → Blank template ROIs generated
2. **MVP** (US1+US2+US3+US4): Preprocessing + VLM integration + Validation → Test independently → Deploy/Demo
3. **Regression Prevention** (US6): Checkbox preservation → Test independently → Deploy/Demo
4. **UX Enhancements** (US5+US7): Visualization + Debug export → Test independently → Deploy/Demo
5. **Polish**: Threshold tuning + Performance validation → Final deploy

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T007)
2. Once Foundational done:
   - Developer A: US1 Blank ROI Cache (T008-T017)
   - Developer B: US2 ROI Preprocessor (T018-T038)
3. After US1 + US2 complete:
   - Developer A: US3 VLM Integration (T039-T052) - requires both US1 and US2
4. After US3 complete:
   - Developer A: US4 Validation Logic (T053-T061)
   - Developer B: US6 Checkbox Preservation (T062-T066)
   - Developer C: US5 Visualization (T067-T074)
   - Developer D: US7 Debug Export (T075-T083)
5. Final: Polish tasks can be distributed (T084-T094)

---

## Notes

- **[P] tasks**: Different files or independent operations, can run in parallel
- **[Story] label**: Maps task to specific user story for traceability
- **File paths**: Exact paths provided for every implementation task
- **MVP scope**: Phases 1-6 deliver core preprocessing functionality (US1-US4)
- **Testing**: Manual validation throughout, pytest for regression testing
- **Environment variable**: DEBUG_ROI_PREPROCESSING=true enables debug image saving
- **Performance target**: <100ms per ROI for preprocessing pipeline
- **Zero regression**: All existing VLM, checkbox, and auxiliary comparison features must continue to work unchanged

---

## Task Count Summary

- **Total Tasks**: 94
- **Phase 1 (Setup)**: 2 tasks
- **Phase 2 (Foundational)**: 5 tasks
- **User Story 1**: 10 tasks
- **User Story 2**: 21 tasks
- **User Story 3**: 14 tasks
- **User Story 4**: 9 tasks
- **User Story 6**: 5 tasks
- **User Story 5**: 8 tasks
- **User Story 7**: 9 tasks
- **Phase 10 (Polish)**: 11 tasks

**Parallelizable Tasks**: 21 tasks marked with [P]

**Critical Path Length**: ~50 tasks (sequential dependencies on critical path)

**Estimated Implementation**: 10-14 hours following incremental strategy, 6-8 hours with parallel team (3-4 developers)
