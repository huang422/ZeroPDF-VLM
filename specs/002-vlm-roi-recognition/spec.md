# Feature Specification: VLM-Based ROI Content Recognition

**Feature Branch**: `002-vlm-roi-recognition`
**Created**: 2025-12-23
**Status**: Draft
**Input**: User description: "VLM 推論 (Zero-Shot Inference) - Add VLM model (InternVL 3.5-1B) to recognize content within extracted ROI regions from aligned documents, including signatures, stamps, checkmarks, and text fields, with structured CSV output"

## Clarifications

### Session 2025-12-23

- Q: How should the system parse VLM responses into structured format? → A: JSON-structured prompting - Append to each prompt: "Respond ONLY with valid JSON: {\"has_content\": true/false, \"content_text\": \"extracted text or null\"}" and parse JSON from response
- Q: What is the CPU quantization preference order when GPU is unavailable? → A: Try INT8 first, if OOM then INT4, if still OOM then warn and run unquantized (may be slow/crash)
- Q: How should the system process multiple ROIs from a single document? → A: Sequential per-ROI - Call model.chat() individually for each ROI in order, simplest implementation
- Q: Should VLM prompts be in English or Chinese for recognizing Chinese document fields? → A: Traditional Chinese prompts (繁體中文) - Use Traditional Chinese instructions for better cultural/linguistic alignment, output text must also be Traditional Chinese
- Q: What is the exact input format for passing ROI images and prompts to InternVL? → A: Use InternVL format: `<image>\n[prompt]`

### Session 2025-12-29

- Q: When VLM recognition encounters exceptions or OOM errors, how should the system restart and retry? → A: Catch exception, release model memory, reload model, then retry current ROI (maximum 3 retries before skipping)
- Q: When a document's `results` field evaluates to False (failed validation), how should the system handle that record? → A: Output normally to CSV with results=False, allowing users to filter non-compliant samples in downstream processing
- Q: How should visualization images display has_content status for each ROI? → A: Color-coded bounding boxes - green for has_content=True, red for has_content=False
- Q: What are the exact field counts for each template? → A: contractor_1 has 13 fields, enterprise_1 has 14 fields, contractor_2 has 2 fields
- Q: When VX1 checkbox is marked (disagreement), should the system continue recognizing other fields? → A: Continue recognizing all fields for completeness, but set results=False immediately when VX1.has_content==True

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Process Documents with VLM Recognition (Priority: P1)

Users process scanned authorization forms and consent documents to automatically extract and validate whether required fields are completed (signatures, stamps, checkmarks, text entries). The system identifies which template each document matches, aligns it, extracts predefined ROI regions, and uses a VLM to determine if each field contains handwritten signatures, official stamps, checkmarks, or filled text.

**Why this priority**: Core value proposition - automated document validation reduces manual review time from hours to minutes and provides structured, queryable results.

**Independent Test**: Upload a contractor authorization form PDF to the input directory, run the processor, verify that the CSV output contains true/false detection results for all required fields (signature presence, stamp presence, checkbox status) plus extracted text content where applicable.

**Acceptance Scenarios**:

1. **Given** a contractor_1 authorization form with all fields completed, **When** the system processes the document, **Then** the output CSV shows `has_content: true` for all signature/stamp/checkbox fields and extracts the filled text (company name, person name, date) with null for any unreadable text.

2. **Given** an enterprise_1 form with missing stamps, **When** the VLM analyzes stamp ROIs, **Then** the CSV shows `has_content: false` for those stamp fields and `content_text: null`.

3. **Given** a multi-page PDF containing contractor_2 forms, **When** batch processing completes, **Then** each page generates a separate CSV row with template type, page number, and all field recognition results.

---

### User Story 2 - Hardware-Adaptive Model Loading (Priority: P2)

The system automatically detects available hardware (GPU/CPU) and loads the InternVL 3.5-1B model with appropriate quantization - full precision on GPU for accuracy, INT4/INT8 quantization on CPU for memory efficiency.

**Why this priority**: Enables deployment across different environments without manual configuration - users with high-end workstations get maximum accuracy while users on standard laptops can still process documents.

**Independent Test**: Run the system on a CPU-only machine and verify that the model loads successfully with quantization (check console logs), processes a test document, and completes without memory errors.

**Acceptance Scenarios**:

1. **Given** a system with CUDA-capable GPU, **When** the processor initializes, **Then** the VLM loads with BF16 or FP16 precision and logs indicate GPU device usage.

2. **Given** a system with only CPU and 8GB+ RAM, **When** the processor initializes, **Then** the VLM loads with INT8 quantization and logs indicate "CPU inference mode, INT8 quantization".

3. **Given** a system with only CPU and <6GB available RAM, **When** INT8 loading triggers OOM, **Then** the system automatically retries with INT4 quantization and logs "Falling back to INT4 quantization due to memory constraints".

4. **Given** a GPU with insufficient VRAM, **When** GPU loading fails, **Then** the system falls back to CPU with INT8 quantization (then INT4 if needed) and logs a warning message.

---

### User Story 3 - Template-Specific Recognition Rules (Priority: P1)

Different document templates (contractor_1, contractor_2, enterprise_1) have different field sets and validation requirements. The system applies template-specific recognition prompts and output schemas to each aligned document based on which template was matched.

**Why this priority**: Business requirement - each form type has legally distinct field requirements (e.g., contractor_1 has 13 fields, contractor_2 has 2 fields, enterprise_1 has 14 fields), and incorrect field mapping would invalidate compliance checks.

**Independent Test**: Process one document of each template type and verify that the output CSV contains exactly the fields defined for that template (contractor_1 has 13 fields including VX1, person1, company1, etc.; contractor_2 has 2 fields: contractor_2_title and small; enterprise_1 has 14 fields).

**Acceptance Scenarios**:

1. **Given** a document matched to contractor_1 template, **When** ROI recognition runs, **Then** the system generates exactly 13 recognition results matching the contractor_1 schema.

2. **Given** a document matched to enterprise_1 template, **When** ROI recognition runs, **Then** the system generates exactly 14 recognition results matching the enterprise_1 schema.

3. **Given** a document matched to contractor_2 template, **When** ROI recognition runs, **Then** the system generates exactly 2 recognition results (title + small stamp).

---

### User Story 4 - Precise Presence Detection with Lenient Content Extraction (Priority: P1)

For each ROI field, the system first determines with high precision whether any content exists (signature stroke, stamp impression, checkbox mark, handwritten/printed text), then attempts to extract readable text. Presence detection must be highly accurate (true/false), while text extraction can gracefully fail to null if illegible.

**Why this priority**: Compliance validation requires knowing definitively whether a field was completed, even if the specific text is unreadable - a signed but illegible signature still indicates agreement.

**Independent Test**: Create test images with edge cases: a very faint stamp, a checkbox with partial mark, barely legible handwriting. Verify that presence detection correctly identifies them as `true` even if content extraction returns `null`.

**Acceptance Scenarios**:

1. **Given** a stamp ROI with a faded red seal (low contrast), **When** the VLM analyzes the region, **Then** the result shows `has_content: true` even if stamp text is unreadable (`content_text: null`).

2. **Given** a checkbox ROI with a light checkmark, **When** the VLM examines the circle, **Then** the result shows `has_content: true` indicating the checkbox was marked.

3. **Given** a text field ROI with messy cursive handwriting, **When** the VLM attempts OCR, **Then** the result shows `has_content: true` and either extracted text or `content_text: null` if illegible.

4. **Given** a stamp ROI with only the printed placeholder text "負責人蓋章處", **When** the VLM analyzes the region, **Then** the result shows `has_content: false` (no actual stamp present).

---

### Edge Cases

- **Multi-page PDFs**: What happens when a PDF contains 50 pages with mixed template types? System should process each page independently and generate CSV rows for all successfully matched pages.

- **Unrecognized templates**: How does system handle ROI recognition when template matching fails? Skip VLM recognition for failed matches and mark as error in CSV (no hallucinated results).

- **Empty/blank ROI regions**: What happens when a required field is completely empty? VLM must return `has_content: false` and `content_text: null` (not hallucinate nonexistent text).

- **Model loading failures**: How does system handle VLM download failures or corrupted model files? Fail gracefully with clear error message indicating model initialization failed, do not crash the entire pipeline.

- **Memory constraints**: What happens on systems with <4GB RAM when processing large batches? Process documents sequentially instead of parallel, and implement model unloading/reloading if needed to prevent OOM crashes.

- **Mixed language content**: How does system handle documents with Chinese/English mixed text? VLM (InternVL 3.5) natively supports multilingual recognition, no special handling needed.

- **Rotated or skewed stamps**: What happens when stamps are applied at angles? Template alignment should normalize orientation before VLM sees the ROI; VLM prompts should specify to detect stamps regardless of rotation within the ROI.

- **Overlapping signatures and stamps**: How does system handle regions where signature overlaps with a stamp? VLM prompt should ask for presence of *either* signature *or* stamp (logical OR for presence detection).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST integrate InternVL 3.5-1B vision-language model into the existing document processing pipeline without disrupting current template matching and alignment functionality.

- **FR-002**: System MUST automatically detect available hardware and load VLM with appropriate precision:
  - GPU available: Load with BF16 or FP16 precision, enable CUDA acceleration
  - CPU only: Attempt INT8 quantization first; if OutOfMemory error occurs, retry with INT4 quantization; if still OOM, log warning and attempt unquantized loading (may be slow or crash on low-memory systems)
  - Log the selected precision level (BF16/FP16/INT8/INT4/unquantized) and device (cuda/cpu) during initialization

- **FR-003**: System MUST accept extracted ROI images and their associated metadata (roi_id, template_id, description) from the existing ROI extraction module.

- **FR-004**: System MUST maintain a mapping of template IDs to their respective field schemas:
  - contractor_1: 13 fields (contractor_1_title, VX1, person1, company1, VX2, company2, company_number1, person2, person_number1, year, month, date, big)
  - contractor_2: 2 fields (contractor_2_title, small)
  - enterprise_1: 14 fields (enterprise_1_title, VX1, person1, company1, VX2, company2, person2, company_number, address, year, month, date, big, small)

- **FR-005**: System MUST generate field-specific VLM prompts in Traditional Chinese for each ROI type with JSON-structured output format:
  - Title fields: No recognition needed, directly output the predefined title string
  - Checkbox fields (VX1, VX2): "仔細檢查這個圓形核取方塊區域。圓圈內是否有任何勾選筆畫或原子筆痕跡？如果您在圓圈邊界內看到任何標記/筆畫，請將 has_content 設為 true；如果圓圈是空的或只顯示印刷外框，請將 has_content 設為 false。僅以有效的 JSON 格式回應：{\"has_content\": true/false, \"content_text\": null}"
  - Stamp fields (small, big, small1, big1, small2, big2): "仔細分析這個印章區域。是否存在實體印章印記（正方形或圓形印章，可能是紅色或黑色墨水）？忽略印刷的預設文字如「負責人蓋章處」。如果您檢測到實際的印章/印記，請將 has_content 設為 true；如果您只看到印刷邊框或預設文字，請將 has_content 設為 false。僅以有效的 JSON 格式回應：{\"has_content\": true/false, \"content_text\": null}"
  - Text fields (person1, company1, person2, company2, person3, address): "檢查這個文字欄位區域。是否填寫了任何手寫或印刷的文字內容？如果您看到任何文字（即使部分難以辨認），請將 has_content 設為 true 並提取繁體中文文字；如果欄位是空白的，請將 has_content 設為 false。僅以有效的 JSON 格式回應：{\"has_content\": true/false, \"content_text\": \"提取的繁體中文文字或 null\"}"
  - Number fields (company_number1, company_number2, person_number1, year, month, date): "檢查這個數字欄位區域。是否存在任何手寫或印刷的數字/數位？如果您看到任何數字內容，請將 has_content 設為 true 並提取數字；如果您只看到印刷邊框/方框，請將 has_content 設為 false。僅以有效的 JSON 格式回應：{\"has_content\": true/false, \"content_text\": \"提取的數字或 null\"}"

- **FR-006**: System MUST process ROI images sequentially using the VLM's single-image chat interface:
  - For each document, iterate through its extracted ROI list in order
  - Convert ROI image (numpy array from OpenCV) to PIL Image in RGB format
  - Preprocess image using InternVL's load_image utility to generate pixel_values tensor
  - Format prompt as: `<image>\n[Traditional Chinese prompt from FR-005]` (image token + newline + prompt text)
  - Call model.chat(tokenizer, pixel_values, question, generation_config) with formatted prompt
  - Wait for response before processing next ROI (sequential, not batched)
  - This simplifies error handling and maintains clear correspondence between ROI and result

- **FR-007**: System MUST parse VLM JSON responses into structured output format for each field:
  - Parse JSON response string using standard JSON decoder
  - Extract has_content: boolean (true/false) - MUST be present for all fields except title fields
  - Extract content_text: string or null - extracted text/number if has_content is true and extraction succeeds, otherwise null
  - If JSON parsing fails, log error and default to has_content=false, content_text=null

- **FR-008**: System MUST handle VLM recognition failures gracefully - if the model cannot determine presence/absence with confidence, default to has_content=false and content_text=null rather than hallucinating.

- **FR-009**: System MUST generate one CSV output file per processed document containing columns:
  - image_ID (unique identifier for the document/page)
  - results (boolean validation status calculated as follows):
    - HIGHEST PRIORITY: If VX1.has_content == True (checkbox marked, indicating disagreement): results = False (system continues recognizing all other fields for data completeness, but final results is predetermined as False)
    - Else if ALL of the following conditions are met: results = True
      - At least one date field (year, month, or date) has has_content == True
      - All other non-title, non-date fields have has_content == True
    - Otherwise: results = False
    - Note: content_text values do NOT affect results calculation; only has_content boolean values matter
    - Implementation note: VX1 check should be performed AFTER all fields are recognized, not during recognition (to maintain complete field data)
  - type (template_id: contractor_1, contractor_2, or enterprise_1)
  - title (the document title extracted from title field)
  - processing_timestamp
  - One pair of columns (field_name_has_content, field_name_content_text) for each field in the template schema
  - All records output to CSV regardless of results value (users filter downstream)

- **FR-010**: System MUST preserve existing pipeline outputs (aligned images, visualization images, ROI cropped images, metadata JSON) while adding VLM recognition results as a new CSV output. Additionally, system MUST enhance visualization images with VLM recognition status:
  - Draw ROI bounding boxes with color-coded borders: Green (BGR: 0,255,0) for has_content=True, Red (BGR: 0,0,255) for has_content=False
  - Maintain existing ROI labels (field names) alongside the colored boxes
  - Use line thickness of 2-3 pixels for clear visibility on document images
  - This allows visual quick-scan of document completion status without opening CSV files

- **FR-011**: System MUST process multi-page PDFs by generating one CSV row per page with page numbers for correlation with existing outputs.

- **FR-012**: System MUST validate that the matched template_id exists in the template-to-schema mapping before attempting VLM recognition, and skip recognition with an error log entry if template is unknown.

- **FR-013**: System MUST implement model caching/singleton pattern to load the VLM model once during processor initialization rather than per-document to avoid repeated loading overhead.

- **FR-014**: For title fields (contractor_1_title, contractor_2_title, enterprise_1_title), system MUST output the predefined title strings directly without VLM inference:
  - contractor_1_title → "企業負責人電信信評報告之使用授權書"
  - contractor_2_title → "個人資料特定目的外用告知事項暨同意書"
  - enterprise_1_title → "企業電信信評報告之使用授權書"

- **FR-015**: System MUST implement robust exception handling for VLM inference failures:
  - Catch all exceptions during model.chat() execution (including OOM errors, CUDA errors, inference timeouts)
  - On exception: Log error details, release model from memory (del model, torch.cuda.empty_cache()), reload model using VLMLoader
  - Retry the same ROI up to 3 times with exponentially increasing delay (1s, 2s, 4s)
  - After 3 failed retries: Skip the ROI, set has_content=false and content_text=null, log warning, continue to next ROI
  - System MUST NOT halt or crash the entire processing pipeline due to individual ROI failures

### Key Entities *(include if feature involves data)*

- **VLMRecognizer**: Component responsible for loading InternVL 3.5-1B model, managing hardware detection, and performing inference on ROI images with field-specific prompts.

- **FieldSchema**: Defines the ordered list of field names for each template type, their recognition types (title/checkbox/stamp/text/number), and associated prompt templates.

- **RecognitionResult**: Structured output for a single field containing has_content (bool), content_text (string|null), and confidence metadata if available.

- **DocumentRecognitionOutput**: Aggregated results for an entire document page, containing document metadata (filename, page, template_id, timestamp) and a list of RecognitionResult objects for all fields.

- **CSVExporter**: Converts DocumentRecognitionOutput objects into CSV format with flattened columns (field_name_has_content, field_name_content_text).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can process a batch of 100 mixed-template documents and receive a single consolidated CSV file with recognition results for all pages in under 10 minutes on a mid-range GPU system (RTX 3060 or equivalent).

- **SC-002**: VLM presence detection (has_content true/false) achieves at least 95% accuracy on test documents with clearly marked/unmarked fields, and at least 85% accuracy on edge cases (faint stamps, light checkmarks).

- **SC-003**: Text extraction success rate (non-null content_text when has_content is true) reaches at least 80% for printed text and at least 60% for handwritten text on legible samples.

- **SC-004**: System successfully processes documents on CPU-only machines with 8GB RAM using quantized models, completing a single-page document in under 30 seconds.

- **SC-005**: Zero existing functionality regression - all current template matching, alignment, and ROI extraction features continue to work identically with VLM module disabled.

- **SC-006**: CSV output files are directly importable into Excel/Google Sheets without format errors, and all boolean/text columns are properly quoted and escaped.

## Assumptions

1. The InternVL 3.5-1B model will be downloaded from HuggingFace (https://huggingface.co/OpenGVLab/InternVL3_5-1B) on first run - users must have internet access during initial setup.

2. VLM inference will add 1-3 seconds per ROI on CPU, 0.1-0.5 seconds per ROI on GPU - batch processing time is acceptable for offline validation workflows (not real-time).

3. Field-specific prompt engineering will be sufficient to achieve target accuracy without fine-tuning the InternVL model - zero-shot learning is adequate.

4. The existing ROI extraction quality is sufficient for VLM recognition - if alignment produces severely distorted ROIs, VLM accuracy will degrade (garbage in, garbage out).

5. Users will validate VLM results for critical compliance use cases - this is an automation aid, not a replacement for human review on legally binding documents.

6. CSV format is sufficient for downstream integration - no need for JSON/XML/database output at this stage.

7. INT4/INT8 quantization provides acceptable accuracy trade-off on CPU (assumed 90%+ preservation of full-precision accuracy based on standard quantization benchmarks).

8. The three template types (contractor_1, contractor_2, enterprise_1) cover all document types in the current workflow - no need to handle unknown template types beyond error logging.

9. VLM prompts and extracted text output will use Traditional Chinese (繁體中文) for better alignment with Taiwan/Hong Kong document standards - the InternVL model supports Traditional Chinese natively.

## Dependencies

- **Existing System Components**: VLM integration depends on the current pipeline's ROI extraction module (vlm_pdf_recognizer.extraction.roi_extractor) to provide cropped ROI images and metadata.

- **External Model**: Requires access to HuggingFace model hub to download InternVL3_5-1B weights (approximately 2-4GB download).

- **Python Libraries**: Assumes PyTorch, Transformers (HuggingFace), and Pillow are available or can be added to the Python 3.9+ vlmcv environment.

- **Hardware Drivers**: GPU inference requires CUDA toolkit and compatible NVIDIA drivers if GPU acceleration is used.

## Out of Scope

- **Model fine-tuning**: This feature uses the pre-trained InternVL 3.5-1B model as-is; custom training or fine-tuning on domain-specific documents is not included.

- **Real-time processing**: The system is designed for batch offline processing; real-time video/camera feed recognition is not supported.

- **Multi-modal prompting**: Only image-based prompts are used; advanced techniques like chain-of-thought reasoning or few-shot learning with example images are not implemented.

- **Result correction UI**: The system outputs CSV results but does not provide a user interface for reviewing/correcting VLM predictions; users must use external tools (Excel, custom scripts) for validation.

- **Template creation**: Adding new document templates beyond contractor_1/2 and enterprise_1 requires code changes to the FieldSchema configuration; self-service template definition is not supported.

- **Language translation**: Content extraction outputs text as-is (Chinese/English mixed); automatic translation to a single language is not provided.
