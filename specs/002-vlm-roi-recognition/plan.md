# Implementation Plan: VLM-Based ROI Content Recognition

**Branch**: `002-vlm-roi-recognition` | **Date**: 2025-12-29 | **Spec**: [spec.md](spec.md)
**Last Aligned With Code**: 2026-05-26
**Status**: Implemented — this plan reflects the production design after the InternVL → Ollama pivot. The original draft (InternVL + JSON prompting + per-doc CSV) is preserved in `research.md` for historical context.

## Summary

Add a recognition + validation + aggregation stage that consumes `ExtractedROI` lists from Feature 001 and produces:

- Per-field `RecognitionResult` (AIP-decided `has_content`, VLM-extracted `content_text`).
- Per-document `DocumentRecognitionOutput` with a boolean `results` from VX1-priority / date-OR / others-AND (including VX2) rules.
- Per-case aggregation gated on (a) all docs valid, (b) all docs target templates, (c) all three required templates present.
- Per-date `VLM_results.json`, flat `vlm_recognition_results.csv`, and `result_log.md` failure summary.
- Colour-coded visualisations overwriting the alignment-stage PNGs.

The VLM is **Ollama-hosted `glm-ocr`** over HTTP. Text is recognised in Traditional Chinese; Simplified is converted post-hoc via OpenCC.

## Technical Context

| Item | Value |
|---|---|
| Language | Python 3.9+ |
| VLM | Ollama `glm-ocr` at `http://localhost:11434` |
| HTTP client | `requests` |
| Image I/O | OpenCV + base64 PNG encoding |
| Chinese conversion | `opencc-python-reimplemented` (s2t) |
| Text cleanup | regex + curated prompt-echo fragment list |
| Hardware | CPU works (Ollama handles GPU offload); GPU recommended for VLM throughput |
| Testing | pytest |
| Throughput | 0.5–1.5 s per ROI on GPU; 3–5 s end-to-end per document |

## Constitution Check

- **Local-first**: ✅ Ollama is local. No cloud calls.
- **Zero training**: ✅ glm-ocr is used as-is.
- **Graceful degradation**: ✅ VLM failures retry 3× then give up with `content_text=None`; whole pipeline can run with `--disable-vlm`.
- **Determinism**: Ollama temperature is 0.0 (`vlm_loader.py`); outputs are reproducible modulo model serving state.

## Major Pivots From the Original Draft

The first draft of this spec assumed **InternVL 3.5-1B** loaded via HuggingFace Transformers, returning **JSON** for every ROI. Three problems forced a pivot:

1. **HF + quantization fragility**: INT4/INT8 fallback was buggy in practice on CPU; cold-start times were minutes.
2. **JSON output unreliable**: small VLMs frequently emitted malformed JSON, especially under prompt echo.
3. **Maintenance burden**: HF stack required PyTorch, Transformers, timm, accelerate, plus model downloads and CUDA-specific deps.

The pivot was to **Ollama + raw OCR text**:

- Ollama owns model lifecycle (pull, quantize, GPU offload). The application talks HTTP.
- `glm-ocr` is an OCR model; raw text is more reliable than asking for JSON.
- `has_content` is now decided by AIP (Feature 004) — a deterministic pixel-difference pipeline — instead of relying on the VLM. The VLM is reduced to "what does this say".
- Validation logic (VX1 priority, date OR, others AND with VX2) was tightened during the pivot.

These changes propagated to: `vlm_loader.py`, `vlm_recognizer.py`, `field_schema.py` (prompts), and `output.py` (case aggregation).

## Project Structure

```text
vlm_pdf_recognizer/
├── recognition/
│   ├── __init__.py
│   ├── vlm_loader.py            # Ollama client singleton + hardware detection
│   ├── vlm_recognizer.py        # VLMRecognizer.process_document / _recognize_field
│   ├── field_schema.py          # FieldSchema, TemplateSchema, PROMPT_TEMPLATES (auto-generated)
│   ├── roi_preprocessor.py      # Owned by Feature 004 (AIP) — imported here
│   └── csv_exporter.py          # Flat CSV writer
└── output.py                    # save_batch_summary_with_vlm, _aggregate_case_results, save_failure_log, save_vlm_visualization

main.py                          # CLI entry point with --disable-vlm flag
```

## Complexity Notes

- **Singleton `VLMLoader`**: required because the Ollama HTTP client is reused across hundreds of ROIs in a batch.
- **Retry with exponential back-off** (1s, 2s, 4s): chosen empirically — `glm-ocr` occasionally returns transient HTTP errors when GPU memory is rebalancing.
- **Prompt-echo detection**: glm-ocr sometimes echoes parts of the prompt back. A curated fragment list discards such outputs rather than trying to parse them.
- **Per-date outputs** (not per-document, not per-case): matches how the team consumes results in batch QA.

## Phase 0 — Research

See [research.md](./research.md). Two distinct research phases:
1. **Original (InternVL)**: kept for historical reference.
2. **Pivot (Ollama)**: dominates current production decisions. Key resolved questions:
   - Prompt language → Traditional Chinese.
   - Output format → raw OCR text (not JSON).
   - `has_content` source → AIP (Feature 004), not VLM.

## Phase 1 — Design

Captured in [data-model.md](./data-model.md). The original draft's `FieldSchema` had 5 field types; production has 7 (added `version` and `person_number`).

## Phase 2 — Tasks

See [tasks.md](./tasks.md). All implementation tasks are complete. New work follows [DEVELOPMENT_WORKFLOW.md](../../DEVELOPMENT_WORKFLOW.md).

## Maintenance Notes

When changing this feature, prefer touching:
- **Prompts**: edit `templates/location/<id>.json` (LabelMe) + re-run `update_configs.py` so `field_schema.PROMPT_TEMPLATES` regenerates. Never hand-edit `field_schema.py`.
- **Cleanup rules** (prompt-echo, stamp placeholders): edit `_clean_content_text` and `_prompt_fragments` in `vlm_recognizer.py`. Update FR-009 in `spec.md`.
- **Validation logic**: edit `calculate_results_status` and `_aggregate_case_results` together. Update FR-012/FR-013 in `spec.md` and the data-model algorithm box.
- **Output schema**: edit `to_json_dict` + `csv_exporter.py` + `save_failure_log` together; update the JSON shape in `data-model.md`.

Any change to `REQUIRED_TEMPLATE_TYPES` (e.g. adding a fourth template type to the case-validity rule) MUST be reflected in `output.py:202`, `spec.md` FR-013, and this plan's "Validation Logic" section.
