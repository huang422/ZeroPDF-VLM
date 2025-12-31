# VLM-pdfRecognizer Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-12-23

## Active Technologies
- File-based - InternVL 3.5-2B model cache (~2-4GB) in HuggingFace cache dir, CSV outputs to output/ (002-vlm-roi-recognition)
- Python 3.9+ (existing vlmcv environment) (004-vlm-roi-preprocessing)

- Python 3.9+ (vlmcv environment) (001-document-template-alignment)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.9+ (vlmcv environment): Follow standard conventions

## Recent Changes
- 004-vlm-roi-preprocessing: Added Python 3.9+ (existing vlmcv environment)
- 003-vlm-auxiliary-roi-comparison: Added Python 3.9+ (vlmcv environment)
- 002-vlm-roi-recognition: Added Python 3.9+ (vlmcv environment)


<!-- MANUAL ADDITIONS START -->

## VLM Recognition Dependencies (002-vlm-roi-recognition)

**Primary Dependencies**:
- PyTorch 2.0+ - Deep learning framework for InternVL model execution
- Transformers 4.52.1+ (HuggingFace) - InternVL model loading and inference API
- Pillow (PIL) - Image format conversion (OpenCV BGR → PIL RGB)
- timm - Vision transformer backbone (InternVL dependency)
- OpenCV (cv2) 4.x - Existing pipeline integration (ROI image input)
- NumPy - Array operations

**Test Commands**:
```bash
# Unit tests (with mocked VLM)
pytest tests/unit/test_vlm_loader.py
pytest tests/unit/test_field_schema.py
pytest tests/unit/test_vlm_recognizer.py
pytest tests/unit/test_csv_exporter.py

# Integration tests (with real InternVL model)
pytest tests/integration/test_vlm_pipeline.py
pytest tests/integration/test_backward_compat.py
```

**Important Notes**:
- All prompts and text outputs must use Traditional Chinese (繁體中文)
- VLM integrated directly into main.py - no separate quickstart
- Model: OpenGVLab/InternVL3_5-2B (auto-downloads ~2-4GB on first run)
- Hardware: GPU (CUDA) preferred, CPU fallback with INT8/INT4 quantization
- VRAM requirements (2B model): BF16: 8GB+, FP16: 6-8GB, INT8: 4-6GB, INT4: <4GB

<!-- MANUAL ADDITIONS END -->

https://huggingface.co/OpenGVLab/InternVL3_5-2B
