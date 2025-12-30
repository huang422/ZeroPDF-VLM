# VLM PDF Recognizer

Local, zero-shot document processing system for Traditional Chinese scanned PDF documents with template-based classification, alignment, ROI extraction, and VLM-powered content recognition.

## Features

### Document Processing
- **Template Classification**: Automatically identifies document type (enterprise_1, contractor_1, contractor_2)
- **Watermark Removal**: HSV thresholding to remove blue, gray, and light red watermarks
- **Geometric Alignment**: SIFT feature matching with homography transformation
- **ROI Extraction**: Extracts predefined regions of interest with bounding box visualization

### VLM Recognition (InternVL 3.5-8B)
- **Zero-Shot Content Recognition**: Detects checkbox marks, stamps, and extracts text content without training
- **Field-Specific Prompts**: Customized Traditional Chinese prompts for different field types
- **Smart Output**: Checkbox/stamp fields output only presence (True/False), text fields include extracted content
- **Validation Logic**: Automatic document validation based on required fields and VX1 disagreement detection
- **Enhanced Model**: 8B parameter model for superior accuracy and reasoning capabilities

### System Features
- **Batch Processing**: Process entire directories with integrated VLM results
- **GPU Acceleration**: Automatic GPU/CPU detection with INT8/INT4 quantization fallback
- **JSON Output**: Structured VLM_results.json with preprocessing and VLM recognition statistics
- **Color-Coded Visualization**: Green boxes for detected content, red boxes for missing content
- **Adaptive Precision**: Automatically selects optimal precision (BF16/FP16/INT8/INT4) based on available VRAM

## Requirements

- Python 3.9+
- OpenCV 4.x
- PyTorch 2.0+
- Transformers 4.52.1+ (HuggingFace)
- **12GB+ VRAM** (GPU mode, 16GB+ recommended for best quality) or **16GB+ RAM** (CPU mode with quantization)

## Installation

```bash
# Clone repository
cd VLM-pdfRecognizer

# Create virtual environment (recommended)
conda create -n vlmcv python=3.9
conda activate vlmcv

# Install dependencies
pip install -r requirements.txt
```

**Note**: InternVL 3.5-8B model (~16GB) will be automatically downloaded from HuggingFace on first run.

## Quick Start

### 1. Directory Structure

```
VLM-pdfRecognizer/
├── templates/
│   ├── images/              # Template images
│   │   ├── enterprise_1.jpg
│   │   ├── contractor_1.jpg
│   │   └── contractor_2.jpg
│   └── location/            # LabelMe ROI annotations
│       ├── enterprise_1.json
│       ├── contractor_1.json
│       └── contractor_2.json
├── data/                    # Auto-generated configs
│   ├── enterprise_1/
│   │   ├── config.json
│   │   └── template_features.pkl (cached)
│   ├── contractor_1/...
│   └── contractor_2/...
├── input/                   # Put your documents here
│   ├── 101.jpg
│   ├── 104.jpg
│   └── ...
└── output/                  # Processing results
```

### 2. Generate Config Files (First Time Only)

```bash
python update_configs.py
```

This reads annotations from `templates/location/*.json` and generates `data/*/config.json`.

### 3. Run Processing

**With VLM content recognition (default)**:
```bash
python main.py
```

**Without VLM (preprocessing only)**:
```bash
python main.py --disable-vlm
```

That's it! The program will:
- Load all templates from `templates/images/`
- Process all files in `input/` directory
- Run VLM recognition on extracted ROIs (if --enable-vlm)
- Save results to `output/` directory

## Output Files

For each processed document, you'll get:

```
output/
├── 101_visualization.png        # Color-coded ROI boxes (green=detected, red=missing)
├── 101_metadata.json            # Processing metadata
├── VLM_results.json             # Integrated preprocessing + VLM results
└── vlm_recognition_results.csv  # VLM recognition results (if VLM enabled)
```

### VLM Results Example (VLM_results.json)

```json
{
  "preprocessing": {
    "total_documents": 10,
    "successful": 10,
    "failed": 0,
    "average_processing_time_ms": 2341.5,
    "template_distribution": {"contractor_1": 5, "contractor_2": 3, "enterprise_1": 2}
  },
  "vlm_recognition": {
    "total_documents": 10,
    "successful": 8,
    "failed": 2,
    "average_processing_time_ms": 15240.3
  },
  "documents": [
    {
      "document_ID": "101.jpg",
      "results": true,
      "type": "contractor_1",
      "title": "企業負責人電信信評報告之使用授權書",
      "processing_timestamp": "2025-12-29T10:30:45",
      "fields": {
        "VX1": {"has_content": false},
        "VX2": {"has_content": false},
        "person1": {"has_content": true, "content_text": "王小明"},
        "company1": {"has_content": true, "content_text": "XX科技股份有限公司"},
        "big": {"has_content": true},
        "small": {"has_content": true}
      }
    }
  ]
}
```

## Performance

- **Preprocessing**: ~2-3 seconds per document (CPU)
- **VLM Recognition**:
  - GPU (RTX 3060): ~0.1-0.5 seconds per ROI
  - CPU with INT8: ~1-3 seconds per ROI
- **Feature Caching**: First run computes SIFT features, subsequent runs are 40% faster

## FlashAttention2 (Optional Optimization)

If you see "FlashAttention2 is not installed" warning, you can safely ignore it. FlashAttention2 is an optional optimization for faster attention computation but is **not required** for the VLM to function correctly.

**Benefits of installing FlashAttention2:**
- 2-4x faster VLM inference speed
- 20-40% lower GPU memory usage
- Does NOT improve recognition accuracy (only speeds up computation)

**Installation requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc) installed on your system
- CUDA_HOME environment variable set

**To install (if you have CUDA Toolkit):**
```bash
pip install flash-attn --no-build-isolation
```

**Note:** If you only have PyTorch with CUDA runtime (no CUDA Toolkit), FlashAttention2 cannot be installed. Your VLM will still work perfectly fine without it.

## License

[Your License Here]
