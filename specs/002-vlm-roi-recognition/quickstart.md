# Quickstart Guide: VLM-Based ROI Content Recognition

**Feature**: 002-vlm-roi-recognition
**Last Aligned With Code**: 2026-05-26
**Status**: Reflects current production behaviour with Ollama-hosted `glm-ocr`.

This guide walks you through exercising the **full** pipeline (Feature 001 alignment + Feature 004 AIP + this feature's VLM + validation + aggregation) end-to-end. For the alignment-only path, see [`specs/001-document-template-alignment/quickstart.md`](../001-document-template-alignment/quickstart.md).

---

## Prerequisites

- Python 3.9+ (`vlmcv` conda env recommended).
- Ollama installed: <https://ollama.com/download>.
- Optional: NVIDIA GPU with ≥ 6 GB VRAM for fast VLM inference. Pipeline runs on CPU but each ROI takes much longer.

---

## One-Time Setup

```bash
# 1. Project
git clone https://github.com/huang422/ZeroPDF-VLM
cd ZeroPDF-VLM
conda create -n vlmcv python=3.9 && conda activate vlmcv
pip install -r requirements.txt

# 2. Generate template configs (only needed once, or after annotations change)
python update_configs.py

# 3. Ollama
ollama serve &                        # Start Ollama in background
ollama pull glm-ocr                   # Pull the OCR model (auto-pulled on first run too)
```

---

## Run the Full Pipeline

```bash
# Place inputs at input/<date>/<case_id>/*.pdf
python main.py
```

To skip VLM entirely (e.g. when debugging alignment): `python main.py --disable-vlm`.

---

## Output Layout

```
output/
└── <date>/
    ├── VLM_results.json                       # Integrated summary
    ├── vlm_recognition_results.csv            # Flat per-field rows
    ├── result_log.md                          # Markdown listing failures only
    └── <case_id>/
        ├── <doc>_visualization.png            # Green / red / blue colour-coded boxes
        ├── metadata/<doc>_metadata.json       # Alignment-stage metadata
        ├── rois/<doc>_roi_<field>.png         # Cropped ROI before AIP
        └── processed_rois/<doc>_roi_<field>_processed.png  # AIP diff image
```

---

## Sanity Check Per Document

For each `<doc>` you process, confirm:

1. **Visualisation** — open `<doc>_visualization.png`:
   - Each ROI has a coloured box. Green = filled, Red = empty, Blue / original-template-colour = title or AIP unavailable.
   - Label format is `<field_id>: True|False|title`.
2. **JSON** — open `VLM_results.json` and find the document under `"documents"`:
   - `template_id` is one of `contractor_1` / `contractor_2` / `enterprise_1`, or `unknown` / `error` for rejected docs.
   - `results: true` only when (a) VX1 is unchecked, (b) at least one of year/month/date is filled, **and** (c) every other required field (including VX2) is filled.
   - `case_results` (added by the aggregator) matches your expectation.
3. **CSV** — open `vlm_recognition_results.csv`:
   - One row per field per document.
   - `case_results` column matches the JSON.

---

## Sanity Check Per Case

A case at `input/<date>/<case_id>/` is **only** valid when:

| Condition | Where it's checked |
|---|---|
| All documents in the case have `results=True`. | `_aggregate_case_results.all_valid` |
| Every document matched a target template (not `"unknown"` / `"error"` / any unrelated PDF). | `_aggregate_case_results.all_targets` |
| All three required template types (`contractor_1`, `contractor_2`, `enterprise_1`) are present in the case. | `_aggregate_case_results.all_types_present` |

If `case_results=False`, check `VLM_results.json → case_results.<case_id>` for `missing_template_types`, `non_target_documents`, and `invalid_count`. The `result_log.md` for that date will also list the diagnosed reason.

---

## Common Diagnostic Patterns

### "Why is this document `results=False`?"

Open `result_log.md` for the date. Each failed document is listed with its diagnosis, in priority order:

1. `非目標文件或 pipeline 失敗 (template_id='unknown')` — Feature 001 rejected the document.
2. `VLM 辨識失敗或前處理錯誤 (無 field_results)` — VLM failure or empty field list.
3. `VX1 (不同意框) 已勾選` — disagreement checkbox checked.
4. `year/month/date 三者皆無內容` — none of the date fields were detected as filled.
5. `以下欄位無內容: [field1, field2, …]` — the AND rule fired; listed fields had `has_content=False`. **Includes VX2 if unchecked.**

### "VLM returned weird text"

- If output contains prompt fragments (`請辨識`, `辨識對象`, …): cleanup discards the response → `content_text=None`. Check `raw_response` for the unfiltered text.
- If output contains stamp placeholder (`負責人蓋章處`): cleanup strips it; if result is empty, `content_text=None`.
- If Simplified Chinese appears: a unit-test for `_clean_content_text` confirms OpenCC is running.

### "Ollama is slow / unresponsive"

- `nvidia-smi` to confirm Ollama is using the GPU.
- `ollama ps` to confirm `glm-ocr` is loaded.
- For a one-shot warm-up: `ollama run glm-ocr "test"` before launching the batch.

---

## Performance Reference

Measured on RTX 4080 Laptop / Ubuntu:

| Stage | Per ROI | Per document |
|---|---|---|
| AIP (Feature 004) | 10–30 ms | depends on field count |
| VLM (this feature) | 500–1500 ms | 2–4 s typically |
| Total end-to-end | — | 3–5 s |

CPU-only is ~10× slower at the VLM stage.

---

## Adding a Field to a Template

1. Re-annotate `templates/location/<template_id>.json` in LabelMe.
2. Run `python update_configs.py` — this regenerates `data/<id>/config.json`, blank ROI references, and `vlm_pdf_recognizer/recognition/field_schema.py`.
3. If the new field has a new `field_type`, also add a prompt template entry to `PROMPT_TEMPLATES` (or `update_configs.py` if you have it generate the prompt).
4. Run a smoke batch and check the new field appears in both `VLM_results.json` and the CSV.

---

## Adding a New Template Type

Affects this feature in two ways:

1. **Field schema**: handled automatically by `update_configs.py`.
2. **`REQUIRED_TEMPLATE_TYPES`**: edit `vlm_pdf_recognizer/output.py:202` to include the new template_id, **and update FR-013 in `spec.md`**. Otherwise cases will fail case-level validation for "missing template type" or "non-target document".

See [DEVELOPMENT_WORKFLOW.md](../../DEVELOPMENT_WORKFLOW.md) for the standard process.
