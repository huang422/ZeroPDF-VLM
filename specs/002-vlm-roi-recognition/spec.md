# Feature Specification: VLM-Based ROI Content Recognition

**Feature Branch**: `002-vlm-roi-recognition`
**Created**: 2025-12-23
**Last Aligned With Code**: 2026-05-26
**Status**: Implemented — this spec reflects current production code in `vlm_pdf_recognizer/recognition/` and `vlm_pdf_recognizer/output.py`.

---

## Scope Recap (Current Implementation)

This feature owns **content recognition + per-document validation + case-level aggregation** of the pipeline:

```
ExtractedROI list (from Feature 001)
  ┬
  ├── AIP (Feature 004) → has_content per field
  └── VLM (this feature) → content_text per field
        │
        ▼
  Per-field RecognitionResult
        │
        ▼
  Per-document validation (VX1 priority, date OR, AND of others incl. VX2)
        │
        ▼
  Per-case aggregation (all docs pass AND all 3 template types present)
        │
        ▼
  Output: VLM_results.json + vlm_recognition_results.csv + result_log.md + colour-coded visualisations
```

> **Major drifts from the original draft of this spec** (kept here so the rewrite is traceable):
> - The draft specified **InternVL 3.5-1B** via HuggingFace Transformers with INT4 / INT8 / unquantized fallback. Production uses **Ollama `glm-ocr`** via HTTP API. Quantization is handled inside Ollama; the application no longer manages it.
> - The draft specified **JSON-structured prompting** (model returns `{"has_content": ..., "content_text": ...}`). Production uses **raw-text OCR**: `glm-ocr` is an OCR-style model that returns text; presence is decided by **AIP** (Feature 004), and VLM is used only for the text content.
> - The draft did not include **VX2-must-be-checked** in the AND validation rule. Production code requires VX2 to be `has_content=True` (the agreement checkbox).
> - The draft did not include **case-level validation** (all three template types must be present per case). Production code enforces this in `output._aggregate_case_results`.
> - The draft used a per-document CSV layout. Production writes **one CSV + one JSON + one Markdown failure log per date** under `output/<date>/`.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Recognise & Validate One Document (Priority: P1)

A user runs the full pipeline on a scanned authorization form. The system already knows which template it matches (Feature 001). This feature must:

1. Detect for each field whether the user filled it in (via AIP — Feature 004).
2. Run VLM OCR on each non-checkbox field that has content to extract Traditional Chinese text / numbers / stamp text.
3. Apply the per-document validation rules to produce `results=True` or `False`.
4. Emit a JSON dict the higher-level orchestrator can aggregate.

**Independent Test**: Place a single PDF whose template = `contractor_1` under `input/<date>/<case_id>/`, run `python main.py` with Ollama serving `glm-ocr`, and confirm:
- `vlm_results` includes one entry with `template_id="contractor_1"`, 13 field results, a boolean `results`, and a `case_id`.
- The visualisation PNG shows green / red / blue boxes per AIP `has_content`.
- The CSV row carries `case_results` (will be False because the case lacks the other two templates).

**Acceptance Scenarios**:

1. **Given** a `contractor_1` PDF with all required fields filled, VX1 unchecked, VX2 checked, year/month/date populated, **When** the pipeline runs, **Then** the per-document `results=True`.
2. **Given** the same PDF but with **VX1 checked** (disagreement), **When** the pipeline runs, **Then** `results=False` regardless of any other field — VX1 has priority. All other fields are still recognised so they appear in JSON / CSV.
3. **Given** a document where AIP says `has_content=False` for `person1`, **When** recognition runs, **Then** VLM is **skipped** for that ROI (cost saving), `content_text=None`, and the AND rule fires → `results=False`.

---

### User Story 2 — Aggregate a Case Across Templates (Priority: P1)

The production workflow batches **cases**: each case has multiple PDFs and must contain exactly the three template types (`contractor_1`, `contractor_2`, `enterprise_1`). The case is valid only when (a) every document passes individual validation **and** (b) all three template types are present **and** (c) no non-target document (e.g. failed match or unrelated PDF) is in the case.

**Independent Test**: Place three PDFs (one per template type) in `input/<date>/<case_id>/`, run the pipeline, and confirm:
- `case_results.<case_id>.case_results == True`.
- `missing_template_types == []`.
- `non_target_documents == []`.

**Acceptance Scenarios**:

1. **Given** a case with all 3 template types and all 3 documents `results=True`, **When** aggregation runs, **Then** `case_results=True` with `missing_template_types=[]`.
2. **Given** a case with only `contractor_1` and `contractor_2` (missing `enterprise_1`), **When** aggregation runs, **Then** `case_results=False` and `missing_template_types=["enterprise_1"]`.
3. **Given** a case containing an unrelated PDF that matched `"unknown"`, **When** aggregation runs, **Then** `case_results=False`, `non_target_documents` lists the offending filename, and the missing-template diagnostic is also populated.
4. **Given** a case with all three template types where one document fails validation, **When** aggregation runs, **Then** `case_results=False` and the failure reason is recorded in `result_log.md` for that date.

---

### User Story 3 — Field-Type-Specific Recognition (Priority: P1)

Different field types need different VLM prompts and post-processing. Production code handles seven types:

| `field_type` | What VLM does | Post-processing |
|---|---|---|
| `title` | Nothing — predefined text is returned. | — |
| `version` | Skipped — `has_content` is always True; text comes from VLM. | Strip non-digit chars. |
| `checkbox` (VX1, VX2) | **Skipped** — `has_content` comes from AIP only. | — (no `content_text` in JSON). |
| `stamp` (big1, small1, etc.) | OCR stamp content. | Strip placeholder text like `負責人蓋章處`; remove whitespace. |
| `text` (person1, company1, address, …) | OCR free-form Chinese text. | Collapse whitespace; convert Simplified → Traditional. |
| `number` (year, month, date, company_number1, …) | OCR digits. | Keep digits only. |
| `person_number` | OCR Taiwan ID format (1 letter + 9 digits). | Keep `[A-Za-z0-9]` only. |

**Independent Test**: Process a populated `contractor_1` form and confirm in `vlm_recognition_results.csv`:
- `contractor_1_title` → predefined string `企業負責人電信信評報告之使用授權書`, no VLM call.
- `VX1` / `VX2` → only `has_content` in JSON, no `content_text`.
- `person_number1` → matches Taiwan ID regex `^[A-Za-z]\d{9}$` (best-effort — VLM may still err).
- `year` / `month` / `date` → digit-only.

---

### User Story 4 — Graceful Degradation on VLM Failures (Priority: P2)

The VLM stage must not crash the batch. Ollama can return errors (HTTP timeout, model not loaded, transient OOM). Production retries up to 3 times with exponential back-off, then logs and continues with `content_text=None`.

**Acceptance Scenarios**:

1. **Given** Ollama returns an exception on first call, **When** the recogniser retries, **Then** the second / third attempts use 1s, 2s, 4s back-off (`time.sleep(2 ** attempt)`).
2. **Given** all 3 attempts fail, **When** the recogniser gives up, **Then** the field gets `content_text=None`, `parse_success=False`, `retry_count=3`, but the document continues processing.
3. **Given** the whole VLM initialisation fails (no Ollama running), **When** the user runs `main.py`, **Then** the pipeline falls back to alignment-only mode and emits a warning (no crash).

---

### User Story 5 — Colour-Coded Visualisation (Priority: P2)

After recognition, the per-document PNG is **overwritten** with colour-coded ROI boxes so a human reviewer can scan a case folder visually.

| Colour | Meaning |
|---|---|
| Green | `has_content=True` (AIP detected content). |
| Red | `has_content=False`. |
| Blue / original | `has_content=None` — title fields, or AIP unavailable. |

**Independent Test**: Open any `<doc>_visualization.png` post-run and confirm at least one green box around populated fields, one red box around empty fields, and the original template colour around the title region.

---

### Edge Cases (Behaviour Today)

| Scenario | Current Behaviour |
|---|---|
| Document fails alignment (Feature 001 returns `success=False`) | A synthetic `DocumentRecognitionOutput(results=False, template_id="unknown" or "error", field_results=[])` is appended; counts toward case "non-target" diagnostics. |
| Ollama returns text that includes the prompt | Cleanup detects known prompt fragments (`請辨識`, `辨識對象`, `格式要求`, …) and discards the output (`content_text=None`). |
| VLM returns Simplified Chinese | Auto-converted via OpenCC (`s2t`) before cleanup. |
| Stamp ROI contains only the printed placeholder `負責人蓋章處` | Cleanup strips known placeholders → `content_text=None`, but AIP should have set `has_content=False`. |
| Multi-page PDF | Each page processed independently; per page `document_ID` becomes `<filename>_page<n>` for `page_number > 0`. |
| Number field that VLM reads as `12月` (text + digit) | Post-processing keeps digits only → `"12"`. |

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST call the VLM via **Ollama HTTP API** with model name `glm-ocr` (default). Ollama base URL is `http://localhost:11434` by default.
- **FR-002**: System MUST auto-detect GPU presence (PyTorch CUDA, falling back to `nvidia-smi` parsing). The detected device is passed to Ollama via `num_gpu=-1` (all available) by default. No application-side INT4 / INT8 fallback — quantization is Ollama's concern.
- **FR-003**: System MUST consume the `extracted_rois: List[ExtractedROI]` produced by Feature 001 and the `template_id` from the same `ProcessingResult`.
- **FR-004**: System MUST keep an in-process registry `TEMPLATE_SCHEMAS` mapping each template ID to a `TemplateSchema`:
  - `contractor_1`: 13 fields — `contractor_1_title`, `VX1`, `person1`, `company1`, `VX2`, `company2`, `company_number1`, `person2`, `person_number1`, `big1`, `year`, `month`, `date`, `version` (14 total; one of them is title).
  - `contractor_2`: 3 fields — `contractor_2_title`, `small1`, `version`.
  - `enterprise_1`: 15 fields — `enterprise_1_title`, `VX1`, `person1`, `company1`, `VX2`, `company2`, `person2`, `company_number1`, `big1`, `small1`, `address`, `year`, `month`, `date`, `version`.
  - Exact list is generated by `update_configs.py` into `vlm_pdf_recognizer/recognition/field_schema.py` from LabelMe annotations — do not hand-edit.
- **FR-005**: System MUST select prompts from `PROMPT_TEMPLATES` (Traditional Chinese) by `field_type`. Prompts must be Traditional Chinese, must instruct the model to **only output the recognised text** (not JSON), and must avoid copying the prompt back into the response.
- **FR-006**: System MUST process ROIs **sequentially** per document — no batching across ROIs in a single Ollama call.
- **FR-007**: For each non-title ROI, system MUST:
  1. Decide `has_content` via AIP (Feature 004) — call `ROIPreprocessor.preprocess_roi(...)` with the document ROI and the blank reference ROI.
  2. **Skip VLM** when (a) `field_type == "checkbox"` or (b) AIP said `has_content=False`. In both cases `content_text=None`.
  3. Otherwise call VLM with the field-type-specific prompt and accept the raw OCR text.
- **FR-008**: System MUST parse the VLM response as **raw OCR text** (empty/whitespace → no content). No JSON decoding.
- **FR-009**: System MUST post-process the VLM text in this order:
  1. Detect & discard prompt-echo fragments (`請辨識`, `觀察重點`, `格式要求`, …) — set `content_text=None` and log a warning.
  2. Strip markdown fences (``` ``` ```) and LaTeX `$$…$$` wrappers.
  3. **OpenCC Simplified → Traditional** conversion.
  4. Remove stamp placeholders (`負責人蓋章處`, `公司大章蓋章處`, etc.).
  5. Apply field-type-specific cleaning:
     - `number` / `version`: keep digits only.
     - `person_number`: keep `[A-Za-z0-9]` only.
     - `stamp` / `text`: collapse whitespace.
- **FR-010**: System MUST retry transient VLM failures up to **3 times** with exponential back-off (1 s, 2 s, 4 s). After exhaustion, set `content_text=None`, `parse_success=False`, `retry_count=3`, log error, and continue.
- **FR-011**: System MUST emit one `RecognitionResult` per field with the schema in [data-model.md](./data-model.md).
- **FR-012**: System MUST emit one `DocumentRecognitionOutput` per page with `results: bool` computed by `calculate_results_status()`:
  1. **VX1 priority**: if any field with `field_id="VX1"` has `has_content=True` → `results=False`.
  2. **Date OR**: among fields with `field_id` in `{year, month, date}`, at least one must have `has_content=True`.
  3. **Other AND**: every remaining field (excluding title, VX1, year/month/date, and any field with `field_type="version"`) must have `has_content=True`. **VX2 is included** in this AND rule — it must be checked.
  4. Final result = `date_valid AND other_fields_valid` (assuming the VX1 priority did not already fire).
- **FR-013**: System MUST aggregate results per **case** in `output._aggregate_case_results`:
  1. Group documents by `case_id`.
  2. `all_valid` = every doc has `results=True`.
  3. `all_targets` = every doc's `template_id` is in `REQUIRED_TEMPLATE_TYPES = {contractor_1, contractor_2, enterprise_1}` (rejects `"unknown"`, `"error"`, or unrelated documents).
  4. `all_types_present` = `{doc.template_id} ⊇ REQUIRED_TEMPLATE_TYPES`.
  5. `case_results` = `all_valid AND all_targets AND all_types_present`.
  6. Also emit `missing_template_types`, `non_target_documents`, `valid_count`, `invalid_count`, `document_count` for diagnostics.
- **FR-014**: System MUST emit per-date outputs under `output/<date>/`:
  - `VLM_results.json`: integrated summary with `preprocessing`, `vlm_recognition`, `case_results`, `documents`.
  - `vlm_recognition_results.csv`: flat per-field rows including `case_results`.
  - `result_log.md`: concise Markdown listing failed documents and failed cases with diagnosed reasons.
- **FR-015**: System MUST overwrite `<doc>_visualization.png` with colour-coded boxes:
  - Green `(0,255,0)` for `has_content=True`.
  - Red `(0,0,255)` for `has_content=False`.
  - Original template colour for `has_content=None` (title fields, AIP unavailable).
  - Label format `"<field_id>: True"` / `"<field_id>: False"` / `"<field_id>: title"`.
- **FR-016**: System MUST save both the **original cropped ROI** (`rois/<base>_roi_<field>.png`) and the **AIP processed diff image** (`processed_rois/<base>_roi_<field>_processed.png`) when VLM is enabled.
- **FR-017**: For title fields, system MUST directly output the predefined string (no VLM call):
  - `contractor_1_title` → `企業負責人電信信評報告之使用授權書`
  - `contractor_2_title` → `個人資料特定目的外用告知事項暨同意書`
  - `enterprise_1_title` → `企業電信信評報告之使用授權書`
  - Authoritative source: `predefined_value` in `field_schema.py`.
- **FR-018**: System MUST be opt-out via `--disable-vlm` so Feature 001 can be exercised alone.
- **FR-019**: `VLMLoader` MUST be a singleton — exactly one Ollama client is created per process and reused across documents.

### Key Entities

See [data-model.md](./data-model.md) for the full field-by-field reference.

| Entity | Module | Role |
|---|---|---|
| `VLMLoader` | `vlm_loader.py` | Singleton; auto-detects hardware, builds Ollama client. |
| `OllamaClient` | `vlm_loader.py` | HTTP wrapper for `/api/generate`. |
| `VLMRecognizer` | `vlm_recognizer.py` | Per-document recognition loop. |
| `FieldSchema` / `TemplateSchema` | `field_schema.py` | Field definitions and prompt templates. Auto-generated. |
| `PROMPT_TEMPLATES` | `field_schema.py` | Dict keyed by `field_type` with Traditional Chinese prompt strings. |
| `RecognitionResult` | `vlm_recognizer.py` | Per-field output (`has_content`, `content_text`, AIP metrics, timing). |
| `DocumentRecognitionOutput` | `vlm_recognizer.py` | Per-page output + `calculate_results_status()`. |
| `_aggregate_case_results` | `output.py` | Case-level boolean & diagnostics. |
| `save_batch_summary_with_vlm` | `output.py` | Writes `VLM_results.json` + invokes CSV / Markdown exporters. |
| `save_failure_log` | `output.py` | Writes `result_log.md` per date. |

---

## Success Criteria *(mandatory)*

Measured on RTX 4080 Laptop / Ubuntu, glm-ocr via Ollama:

- **SC-001**: Per-ROI VLM call completes in **0.5–1.5 s** on GPU.
- **SC-002**: Total per-document time (alignment + AIP + VLM) is **3–5 s**.
- **SC-003**: AIP-driven `has_content` accuracy on populated fields (see Feature 004 SCs).
- **SC-004**: When `--disable-vlm` is set, the application starts and processes documents with **zero** dependency on Ollama / OpenCC.
- **SC-005**: A case that has all three template types and all documents pass validation is consistently marked `case_results=True`. A case missing any template type is consistently marked `False` with `missing_template_types` populated.
- **SC-006**: Prompt-echo detection successfully discards any VLM output containing a known prompt fragment from `_prompt_fragments`.
- **SC-007**: The Simplified → Traditional Chinese conversion runs on every non-empty VLM output (verifiable by checking output text is in Traditional Chinese characters).

---

## Assumptions

1. **Ollama is reachable at `localhost:11434`** with `glm-ocr` pulled. If not, the user sees a startup warning and the pipeline falls back to alignment-only mode.
2. **`update_configs.py` has been run** so that `field_schema.py` is in sync with the LabelMe annotations.
3. **AIP (Feature 004) is the authority for `has_content`** of all non-title, non-version fields. VLM `content_text` is decoupled from `has_content`.
4. **`glm-ocr` returns Traditional Chinese (or Simplified that converts cleanly via OpenCC)**. The cleanup step assumes this — non-Chinese outputs go through whitespace-collapse only.
5. **Per-document validation is binary**. Partial credit / weighting is not supported.
6. **The three template types are the entire universe.** Any document outside them is "non-target" and fails its case.

---

## Out of Scope

- Fine-tuning / training a custom OCR model.
- Returning structured JSON from the VLM (the original draft's approach — abandoned).
- Per-document CSV (now per-date).
- Cloud / hosted VLM. Ollama is local-only by design.
- User-facing review UI. Reviewers consume `VLM_results.json`, the CSV, and `result_log.md`.
- Multilingual prompts. All prompts are Traditional Chinese.
- Adaptive prompt selection / chain-of-thought / few-shot. Plain prompts only.

---

## Dependencies

- **Ollama** running locally with `glm-ocr` available (auto-pull on first run if missing — see `vlm_loader.py`).
- **`opencc-python-reimplemented`** — Simplified → Traditional Chinese conversion (singleton `OpenCC('s2t')`).
- **`requests`** — Ollama HTTP client.
- **Feature 001** — alignment + ROI extraction (already covered by `vlm_pdf_recognizer/pipeline.py`).
- **Feature 004** — AIP for `has_content` detection (`roi_preprocessor.py` + `BlankTemplateROICache`).

---

## Notes for Spec-vs-Code Reviewers

- The CSV is **flat**: one row per field, with `case_results` duplicated across rows of the same case. See `csv_exporter.py`.
- The visualisation written by Feature 001 is overwritten in-place by `save_vlm_visualization` — this is intentional (single file per document).
- `RecognitionResult` carries both AIP and VLM metrics in dedicated fields (`AIP_has_content`, `AIP_ink_ratio`, `AIP_component_count`, `AIP_time_ms`) for debugging. These are **not exposed** in `to_json_dict()` — only `has_content` and `content_text` appear in the JSON output by design.
- `version` field text is extracted by VLM but `has_content` is hard-wired to `True` and the field is excluded from validation (FR-012 step 3).
