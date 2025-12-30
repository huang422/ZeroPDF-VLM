# Tasks: VLM-Based ROI Content Recognition

**Input**: Design documents from `/specs/002-vlm-roi-recognition/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

**Tests**: No test tasks included - tests not explicitly requested in specification. Focus on implementation and integration.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

Project structure (from plan.md):
- Source code: `vlm_pdf_recognizer/` (library modules)
- Tests: `tests/unit/`, `tests/integration/`
- Data: `data/` (template definitions)
- Output: `output/` (CSV results)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create new VLM recognition module structure

- [ ] T001 Create recognition module directory structure at vlm_pdf_recognizer/recognition/ with __init__.py
- [ ] T002 [P] Create test directory structure at tests/unit/ for VLM tests
- [ ] T003 [P] Create integration test directory at tests/integration/ for end-to-end VLM tests
- [ ] T004 [P] Update requirements.txt with PyTorch 2.0+, transformers 4.52.1+, Pillow, timm, pandas dependencies

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core VLM infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Implement VLMConfig dataclass in vlm_pdf_recognizer/recognition/vlm_loader.py with hardware detection (torch.cuda.is_available(), vram/ram detection via psutil)
- [ ] T006 Implement FieldSchema dataclass in vlm_pdf_recognizer/recognition/field_schema.py with validation (field_type validation, prompt_template format checking)
- [ ] T007 Implement TemplateSchema dataclass in vlm_pdf_recognizer/recognition/field_schema.py with field_schemas list and validation
- [ ] T008 Implement RecognitionResult dataclass in vlm_pdf_recognizer/recognition/vlm_recognizer.py with has_content, content_text, raw_response, parse_success fields
- [ ] T009 Implement DocumentRecognitionOutput dataclass in vlm_pdf_recognizer/recognition/vlm_recognizer.py with document metadata, field_results list, and to_csv_row() method
- [ ] T010 Define Traditional Chinese prompt templates (checkbox, stamp, text, number) in vlm_pdf_recognizer/recognition/field_schema.py as PROMPT_TEMPLATES dict
- [ ] T011 Define template-to-field mappings for contractor_1 (13 fields), contractor_2 (2 fields), enterprise_1 (14 fields) in vlm_pdf_recognizer/recognition/field_schema.py as TEMPLATE_SCHEMAS dict

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 2 - Hardware-Adaptive Model Loading (Priority: P2) 🎯 MVP

**Goal**: System automatically detects hardware (GPU/CPU) and loads InternVL 3.5-1B with appropriate quantization (BF16/FP16 on GPU, INT8→INT4 fallback on CPU)

**Independent Test**: Run on CPU-only machine, verify model loads with INT8 quantization (check console logs), processes a test document, completes without memory errors

**Why User Story 2 First**: Hardware detection is a prerequisite for all VLM operations. Must be implemented before any recognition can occur.

### Implementation for User Story 2

- [ ] T012 [US2] Implement detect_device_and_load_model() function in vlm_pdf_recognizer/recognition/vlm_loader.py with torch.cuda.is_available() check
- [ ] T013 [US2] Implement GPU model loading path (BF16 precision, use_flash_attn=True) with VRAM detection in vlm_pdf_recognizer/recognition/vlm_loader.py
- [ ] T014 [US2] Implement CPU INT8 quantization loading path (load_in_8bit=True, BitsAndBytes) with RAM detection in vlm_pdf_recognizer/recognition/vlm_loader.py
- [ ] T015 [US2] Implement CPU INT4 fallback on OOM (load_in_4bit=True) in vlm_pdf_recognizer/recognition/vlm_loader.py
- [ ] T016 [US2] Implement unquantized fallback with warning on INT4 OOM in vlm_pdf_recognizer/recognition/vlm_loader.py
- [ ] T017 [US2] Implement GPU→CPU fallback on GPU OOM in vlm_pdf_recognizer/recognition/vlm_loader.py
- [ ] T018 [US2] Add structured logging for device, precision, VRAM/RAM in vlm_pdf_recognizer/recognition/vlm_loader.py
- [ ] T019 [US2] Implement tokenizer loading (AutoTokenizer.from_pretrained with trust_remote_code=True) in vlm_pdf_recognizer/recognition/vlm_loader.py
- [ ] T020 [US2] Create VLMLoader class with singleton pattern to cache loaded model in vlm_pdf_recognizer/recognition/vlm_loader.py with get_instance() method

**Checkpoint**: At this point, User Story 2 should be fully functional - model loads successfully on any hardware

---

## Phase 4: User Story 1 - Process Documents with VLM Recognition (Priority: P1)

**Goal**: Users process documents to extract and validate field completion (signatures, stamps, checkmarks, text). System matches template, aligns, extracts ROIs, and uses VLM to determine field content.

**Independent Test**: Place test PDFs (contractor_1, contractor_2, enterprise_1) in input/ directory, run `python main.py`, verify CSV output in output/ directory contains recognition results with has_content and content_text columns for all fields.

**Why After User Story 2**: Requires model loading capability from US2. This is the core value proposition.

### Implementation for User Story 1

- [ ] T021 [US1] Implement preprocess_roi_for_vlm() function in vlm_pdf_recognizer/recognition/vlm_recognizer.py to convert OpenCV BGR numpy array → PIL RGB → InternVL pixel_values tensor using model.load_image()
- [ ] T022 [US1] Implement parse_vlm_response() function in vlm_pdf_recognizer/recognition/vlm_recognizer.py with json.loads() to extract has_content and content_text from JSON response, with try-except fallback to default values
- [ ] T023 [US1] Implement recognize_single_field() function in vlm_pdf_recognizer/recognition/vlm_recognizer.py that takes (roi_image, field_schema, model, tokenizer) and returns RecognitionResult
- [ ] T024 [US1] Implement title field special handling (skip VLM, output predefined_value) in recognize_single_field() in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T025 [US1] Implement VLM inference call (model.chat() with generation_config: max_new_tokens=256, do_sample=False, temperature=0.0) in recognize_single_field() in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T026 [US1] Implement recognize_document_fields() function in vlm_pdf_recognizer/recognition/vlm_recognizer.py for sequential per-ROI processing, returning List[RecognitionResult]
- [ ] T027 [US1] Implement VLMRecognizer class in vlm_pdf_recognizer/recognition/vlm_recognizer.py with __init__(model, tokenizer, template_schemas) and process_document(roi_images, template_id) method returning DocumentRecognitionOutput
- [ ] T028 [US1] Implement export_recognition_results_to_csv() function in vlm_pdf_recognizer/recognition/csv_exporter.py using pandas DataFrame with flattened columns (field_id_has_content, field_id_content_text)
- [ ] T029 [US1] Update CSV export to use UTF-8-sig encoding and QUOTE_NONNUMERIC quoting in vlm_pdf_recognizer/recognition/csv_exporter.py for Excel compatibility
- [ ] T030 [US1] Add graceful error handling for VLM recognition failures (log error, default to has_content=false) in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T031 [US1] Integrate VLMRecognizer into existing pipeline in vlm_pdf_recognizer/pipeline.py by adding optional VLM recognition step after ROI extraction
- [ ] T032 [US1] Update main.py to initialize VLMLoader, create VLMRecognizer instance, and call recognition for each processed document
- [ ] T033 [US1] Update main.py to call CSV export after each document processing
- [ ] T034 [US1] Add enable_vlm_recognition flag/config option in main.py for backward compatibility

**Checkpoint**: At this point, User Story 1 should be fully functional - end-to-end document processing with VLM recognition and CSV output

---

## Phase 5: User Story 3 - Template-Specific Recognition Rules (Priority: P1)

**Goal**: System applies template-specific recognition prompts and output schemas to each document based on matched template (contractor_1: 13 fields, contractor_2: 2 fields, enterprise_1: 14 fields).

**Independent Test**: Process one document of each template type (contractor_1, contractor_2, enterprise_1), verify CSV output has correct number of field columns matching template schema, verify prompts are template-specific.

**Why After User Story 1**: Extends the core recognition pipeline with template-specific logic.

### Implementation for User Story 3

- [ ] T035 [US3] Define CONTRACTOR_1_FIELDS list in vlm_pdf_recognizer/recognition/field_schema.py with 13 FieldSchema objects (contractor_1_title, VX1, person1, company1, VX2, company2, company_number1, person2, person_number1, year, month, date, big)
- [ ] T036 [US3] Define CONTRACTOR_2_FIELDS list in vlm_pdf_recognizer/recognition/field_schema.py with 2 FieldSchema objects (contractor_2_title, small)
- [ ] T037 [US3] Define ENTERPRISE_1_FIELDS list in vlm_pdf_recognizer/recognition/field_schema.py with 14 FieldSchema objects (enterprise_1_title, VX1, person1, company1, VX2, company2, person2, company_number, address, year, month, date, big, small)
- [ ] T038 [US3] Populate TEMPLATE_SCHEMAS dict with {contractor_1: TemplateSchema, contractor_2: TemplateSchema, enterprise_1: TemplateSchema} in vlm_pdf_recognizer/recognition/field_schema.py
- [ ] T039 [US3] Implement template validation in VLMRecognizer.process_document() to check template_id exists in TEMPLATE_SCHEMAS, skip with error log if unknown
- [ ] T040 [US3] Add template_id column to CSV output in csv_exporter.py
- [ ] T041 [US3] Validate CSV column count matches template field count in export_recognition_results_to_csv() in vlm_pdf_recognizer/recognition/csv_exporter.py

**Checkpoint**: At this point, User Story 3 should be fully functional - all three templates recognized with correct field schemas

---

## Phase 6: User Story 4 - Precise Presence Detection with Lenient Content Extraction (Priority: P1)

**Goal**: System determines with high precision whether content exists (has_content true/false) and attempts text extraction (content_text string or null if unreadable).

**Independent Test**: Create test images with edge cases: faint stamp, partial checkmark, barely legible handwriting. Verify presence detection correctly identifies them as `true` even if content extraction returns `null`.

**Why After User Story 3**: Requires template-specific prompts from US3. Focuses on improving accuracy of existing recognition.

### Implementation for User Story 4

- [ ] T042 [US4] Enhance checkbox prompts in PROMPT_TEMPLATES to emphasize presence detection (look for ANY mark/stroke within circle boundary) in vlm_pdf_recognizer/recognition/field_schema.py
- [ ] T043 [US4] Enhance stamp prompts in PROMPT_TEMPLATES to detect faded/low-contrast stamps and ignore placeholder text in vlm_pdf_recognizer/recognition/field_schema.py
- [ ] T044 [US4] Enhance text prompts in PROMPT_TEMPLATES to detect handwritten text even if illegible in vlm_pdf_recognizer/recognition/field_schema.py
- [ ] T045 [US4] Update parse_vlm_response() to prioritize has_content over content_text (allow has_content=true with content_text=null) in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T046 [US4] Add validation logic to ensure has_content=true is returned for any visible mark/content, regardless of readability in vlm_pdf_recognizer/recognition/vlm_recognizer.py

**Checkpoint**: At this point, User Story 4 should be fully functional - presence detection highly accurate, content extraction gracefully fails to null

---

## Phase 7: Additional Features

**Purpose**: Multi-page support and advanced validation logic

### Multi-Page PDF Support

- [ ] T047 Add page_number parameter to VLMRecognizer.process_document() in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T048 Update DocumentRecognitionOutput to include page_number field in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T049 Update CSV export to include page_number column in vlm_pdf_recognizer/recognition/csv_exporter.py
- [ ] T050 Update main.py to iterate through multi-page PDFs and call VLM recognition per page with correct page_number
- [ ] T051 Update CSV export to append rows for multi-page documents (single CSV with multiple rows) in main.py

### Document Validation Logic (results field)

- [ ] T052 Implement compute_results_status() method in DocumentRecognitionOutput in vlm_pdf_recognizer/recognition/vlm_recognizer.py to calculate validation status based on has_content fields
- [ ] T053 Implement VX1 priority check (if VX1.has_content==True → results=False) in compute_results_status() in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T054 Implement date field logic (at least one of year/month/date must have has_content==True) in compute_results_status() in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T055 Implement all-other-fields check (all non-title non-date fields must have has_content==True) in compute_results_status() in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T056 Add results column to CSV output showing final validation status in vlm_pdf_recognizer/recognition/csv_exporter.py

### Visualization Enhancements

- [ ] T057 [P] Update draw_roi_boxes() function in vlm_pdf_recognizer/output.py to accept optional recognition_results parameter
- [ ] T058 [P] Implement color-coded ROI boxes in output.py: Green (BGR: 0,255,0) for has_content=True, Red (BGR: 0,0,255) for has_content=False
- [ ] T059 [P] Maintain existing ROI labels alongside color-coded boxes in output.py
- [ ] T060 [P] Use line thickness of 2-3 pixels for clear visibility in output.py

### Exception Handling & Retry Logic (FR-015)

- [ ] T061 Implement exception catching in recognize_single_field() for model.chat() execution in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T062 Implement model unloading (del model, torch.cuda.empty_cache()) on exception in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T063 Implement model reloading via VLMLoader.get_instance(force_reload=True) on exception in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T064 Implement retry logic with exponential backoff (1s, 2s, 4s) for up to 3 retries in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T065 Implement skip-on-max-retries with has_content=False, content_text=null, error_message logged in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T066 Add retry_count field to RecognitionResult in vlm_pdf_recognizer/recognition/vlm_recognizer.py

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T067 [P] Add inference_time_ms tracking to RecognitionResult in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T068 [P] Add total_processing_time_ms to DocumentRecognitionOutput in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T069 [P] Add timestamp field to DocumentRecognitionOutput in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T070 [P] Add processing_timestamp column to CSV export in vlm_pdf_recognizer/recognition/csv_exporter.py
- [ ] T071 [P] Update vlm_pdf_recognizer/recognition/__init__.py to export public classes (VLMLoader, VLMRecognizer, FieldSchema, etc.)
- [ ] T072 Add comprehensive error handling for model loading failures in vlm_pdf_recognizer/recognition/vlm_loader.py
- [ ] T073 Add comprehensive error handling for JSON parsing failures in vlm_pdf_recognizer/recognition/vlm_recognizer.py
- [ ] T074 Code cleanup: Ensure all Traditional Chinese prompts use 繁體中文 in vlm_pdf_recognizer/recognition/field_schema.py
- [ ] T075 Validate backward compatibility: Ensure existing pipeline tests pass without VLM enabled
- [ ] T076 [P] Add docstrings to all public functions and classes in vlm_pdf_recognizer/recognition/
- [ ] T077 Update CLAUDE.md with VLM-specific dependencies and test commands (already completed)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 2 - Hardware Loading (Phase 3)**: Depends on Foundational phase - BLOCKS User Story 1
- **User Story 1 - Core Recognition (Phase 4)**: Depends on User Story 2 completion
- **User Story 3 - Template Rules (Phase 5)**: Depends on User Story 1 completion
- **User Story 4 - Presence Detection (Phase 6)**: Depends on User Story 3 completion
- **Additional Features (Phase 7)**: Depends on all user stories being complete
- **Polish (Phase 8)**: Depends on all features being complete

### User Story Dependencies

- **User Story 2 (P2)**: Hardware-Adaptive Model Loading - MUST complete first (blocks all other stories)
- **User Story 1 (P1)**: Process Documents with VLM Recognition - Requires US2, core functionality
- **User Story 3 (P1)**: Template-Specific Recognition Rules - Requires US1, extends with template logic
- **User Story 4 (P1)**: Precise Presence Detection - Requires US3, improves accuracy

### Within Each User Story

- Dataclasses before services
- Services before integration
- Core implementation before error handling
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1 (Setup)**: All T001-T004 can run in parallel [P] - different directories
- **Phase 2 (Foundational)**: T006-T007 (field_schema.py) can run in parallel, T008-T009 (vlm_recognizer.py) can run in parallel
- **Phase 7 (Visualization)**: T057-T060 can run in parallel [P] - different concerns
- **Phase 8 (Polish)**: T067-T071 can run in parallel [P] - different concerns
- User stories CANNOT run in parallel due to sequential dependencies (US2→US1→US3→US4)

---

## Parallel Example: Phase 1 Setup

```bash
# Launch all setup tasks together:
Task: "Create recognition module directory structure at vlm_pdf_recognizer/recognition/"
Task: "Create test directory structure at tests/unit/"
Task: "Create integration test directory at tests/integration/"
Task: "Update requirements.txt with dependencies"
```

## Parallel Example: Phase 8 Polish

```bash
# Launch all performance tracking tasks together:
Task: "Add inference_time_ms tracking to RecognitionResult"
Task: "Add total_processing_time_ms to DocumentRecognitionOutput"
Task: "Add timestamp field to DocumentRecognitionOutput"
Task: "Add processing_timestamp column to CSV export"
Task: "Update __init__.py to export public classes"
```

---

## Implementation Strategy

### MVP First (User Stories 2 + 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 2 (Hardware Loading)
4. Complete Phase 4: User Story 1 (Core Recognition)
5. **STOP and VALIDATE**: Test end-to-end processing with VLM recognition
6. Deploy/demo if ready - **MVP delivers core document processing with VLM recognition**

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 2 (Hardware Loading) → Test independently → Model loads on any hardware
3. Add User Story 1 (Core Recognition) → Test independently → Deploy/Demo (MVP!)
4. Add User Story 3 (Template Rules) → Test independently → Deploy/Demo (all 3 templates)
5. Add User Story 4 (Presence Detection) → Test independently → Deploy/Demo (improved accuracy)
6. Add Phase 7 features → Test independently → Deploy/Demo (full feature set)
7. Each story adds value without breaking previous stories

### Single Developer Strategy

Sequential implementation in dependency order:
1. Setup (Phase 1) → ~1 hour
2. Foundational (Phase 2) → ~3-4 hours (dataclasses, template definitions)
3. User Story 2 (Phase 3) → ~2-3 hours (hardware detection, model loading)
4. User Story 1 (Phase 4) → ~4-6 hours (ROI preprocessing, VLM inference, CSV export, integration)
5. User Story 3 (Phase 5) → ~2-3 hours (template-specific schemas)
6. User Story 4 (Phase 6) → ~1-2 hours (prompt refinement)
7. Phase 7 (Additional Features) → ~3-4 hours (multi-page, validation, visualization, retry logic)
8. Polish (Phase 8) → ~2-3 hours (error handling, performance tracking, documentation)

**Total estimated effort**: 18-26 hours for complete implementation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- User Story 2 (Hardware Loading) is **prerequisite** for User Story 1 despite both being P1/P2 priority
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All prompts must use Traditional Chinese (繁體中文)
- Backward compatibility maintained via optional enable_vlm_recognition flag
- Model caches at ~/.cache/huggingface/ (~2-4GB download on first run)
