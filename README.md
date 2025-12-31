# VLM PDF Recognizer

Local, zero-shot document processing system for Traditional Chinese scanned PDF documents with template-based classification, alignment, ROI extraction, and VLM-powered content recognition.

## Features

### Document Processing
- **Template Classification**: Automatically identifies document type (enterprise_1, contractor_1, contractor_2)
- **Watermark Removal**: HSV thresholding to remove blue, gray, and light red watermarks
- **Geometric Alignment**: SIFT feature matching with homography transformation
- **ROI Extraction**: Extracts predefined regions of interest with bounding box visualization

### AIP (Advanced Image Processing) - ROI Content Detection
- **Template Difference**: Direct BGR pixel difference with ECC sub-pixel alignment
- **Multi-Stage Thresholding**: Adaptive threshold for pre-printed text handling
- **Fast Detection**: ~10-30ms per ROI (template difference-based)
- **High Accuracy**: 100% detection rate on test set (26 ROIs)
- **Pre-VLM Filtering**: Skips VLM inference for confirmed empty fields, reducing processing time

### VLM Recognition (InternVLM 3.5-2B)
- **Zero-Shot Content Recognition**: Detects checkbox marks, stamps, and extracts text content without training
- **Field-Specific Prompts**: Customized Traditional Chinese prompts for different field types
- **Smart Output**: Checkbox/stamp fields output only presence (True/False), text fields include extracted content
- **Validation Logic**: Automatic document validation based on required fields and VX disagreement detection
- **Compact Model**: 2B parameter model for efficient processing

### System Features
- **Dual Detection**: AIP (template difference) + VLM (content recognition)
- **Batch Processing**: Process entire directories with integrated results
- **GPU Acceleration**: Automatic GPU/CPU detection with INT8/INT4 quantization fallback
- **JSON Output**: Structured VLM_results.json with AIP and VLM recognition statistics
- **Color-Coded Visualization**:
  - Green boxes: Content detected (field_id: True)
  - Red boxes: No content (field_id: False)
  - Blue boxes: AIP error (field_id: ERROR)
- **Adaptive Precision**: Automatically selects optimal precision (BF16/FP16/INT8/INT4) based on available VRAM

## Requirements

- Python 3.9+
- OpenCV 4.x
- PyTorch 2.0+ (for VLM only)
- Transformers 4.52.1+ (HuggingFace, for VLM only)
- **6-8GB VRAM** (GPU mode with FP16) or **16GB+ RAM** (CPU mode with INT8 quantization)

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

**Note**: InternVL 3.5-2B model (~4-6GB) will be automatically downloaded from HuggingFace on first run.

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
│   │   ├── template_features.pkl (cached)
│   │   └── blank_rois/          # Blank ROI reference images for AIP
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

This script:
- Reads annotations from `templates/location/*.json`
- Generates `data/*/config.json` for each template
- Extracts blank ROI images for AIP template difference

### 3. Run Processing

**With VLM content recognition (default)**:
```bash
python main.py
```

**Without VLM (AIP only)**:
```bash
python main.py --disable-vlm
```

That's it! The program will:
- Load all templates from `templates/images/`
- Process all files in `input/` directory
- Run AIP (template difference) on all ROIs
- Run VLM recognition on extracted ROIs (if enabled)
- Save results to `output/` directory

## Output Files

For each processed document, you'll get:

```
output/
├── 101_visualization.png        # Color-coded ROI boxes
│                                 # Green: field_id: True
│                                 # Red: field_id: False
│                                 # Blue: field_id: ERROR
├── 101_metadata.json            # Processing metadata
├── rois/                        # Original extracted ROI images
│   ├── 101_roi_person1.png
│   └── ...
├── processed_rois/              # AIP-processed ROI images (difference)
│   ├── 101_roi_person1_processed.png
│   └── ...
├── VLM_results.json             # Integrated AIP + VLM results
└── vlm_recognition_results.csv  # CSV export (if VLM enabled)
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
    "average_processing_time_ms": 4520.3
  },
  "documents": [
    {
      "document_ID": "101.jpg",
      "results": true,
      "type": "contractor_1",
      "title": "企業負責人電信信評報告之使用授權書",
      "processing_timestamp": "2025-12-31T15:30:45",
      "fields": {
        "VX1": {
          "VLM_has_content": false,
          "AIP_has_content": false
        },
        "VX2": {
          "VLM_has_content": false,
          "AIP_has_content": false
        },
        "person1": {
          "VLM_has_content": true,
          "content_text": "王小明",
          "AIP_has_content": true
        },
        "company1": {
          "VLM_has_content": true,
          "content_text": "XX科技股份有限公司",
          "AIP_has_content": true
        },
        "big1": {
          "VLM_has_content": true,
          "AIP_has_content": true
        },
        "small1": {
          "VLM_has_content": true,
          "AIP_has_content": true
        }
      }
    }
  ]
}
```

### CSV Export Format

When VLM is enabled, results are also exported to `vlm_recognition_results.csv`:

```csv
document_ID,page_number,template_id,results,processing_timestamp,VX1_VLM_has_content,VX1_AIP_has_content,person1_VLM_has_content,person1_content_text,person1_AIP_has_content,...
101.jpg,0,contractor_1,True,2025-12-31T15:30:45,False,False,True,王小明,True,...
```

## Performance

- **Document Alignment**: ~1-2 seconds per document (SIFT matching + homography)
- **AIP (Template Difference)**: ~10-30ms per ROI
- **VLM Recognition** (2B model):
  - GPU (RTX 3060, FP16): ~0.5-1.5 seconds per ROI
  - GPU (RTX 3060, INT8): ~0.3-0.8 seconds per ROI
  - CPU (INT8): ~2-4 seconds per ROI
- **Overall Processing**: ~3-5 seconds per document (with VLM)

## AIP Detection Method

The current AIP uses a simple and effective template difference approach:

1. **ECC Alignment**: Sub-pixel alignment of document ROI to blank template ROI
2. **BGR Difference**: Absolute pixel difference across all color channels
3. **Multi-Stage Thresholding**:
   - Calculate mean difference (0.0-1.0)
   - For high difference (>0.15): Use significant pixel ratio (>20% with diff>30)
   - For normal fields: Use mean difference threshold (>0.01)

This method achieves **100% accuracy** on test set with **54% less code** compared to previous complex pipeline.

### Configuration

Only one parameter needs adjustment in `vlm_pdf_recognizer/recognition/config.py`:

```python
MIN_ABSOLUTE_DENSITY_THRESHOLD = 0.01  # Main threshold for normal fields
```

For detailed configuration guide, see [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

## Visualization Legend

Output visualizations (`*_visualization.png`) use color-coded bounding boxes:

- **Green boxes**: Content detected
  - Label format: `field_id: True`
- **Red boxes**: No content detected
  - Label format: `field_id: False`
- **Blue boxes**: AIP error
  - Label format: `field_id: ERROR`

## FlashAttention2 (Optional Optimization)

If you see "FlashAttention2 is not installed" warning, you can safely ignore it. FlashAttention2 is an optional optimization for faster attention computation but is **not required** for the VLM to function correctly.

**Benefits of installing FlashAttention2:**
- 2-4x faster VLM inference speed
- 20-40% lower GPU memory usage
- Does NOT improve recognition accuracy (only speeds up computation)

**Installation requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc) installed on your system

**To install (if you have CUDA Toolkit):**
```bash
pip install flash-attn --no-build-isolation
```

## Project Structure

```
vlm_pdf_recognizer/
├── alignment/              # Document alignment (SIFT, homography)
├── extraction/             # ROI extraction
├── preprocessing/          # PDF to image conversion
├── recognition/            # AIP + VLM recognition
│   ├── config.py           # Configuration (only 4 active parameters)
│   ├── roi_preprocessor.py # AIP processor (391 lines, -54%)
│   ├── vlm_recognizer.py   # VLM recognition
│   ├── vlm_loader.py       # Model loading
│   ├── field_schema.py     # Field definitions
│   └── csv_exporter.py     # CSV export
├── templates/              # Template management
├── output.py               # Result saving
└── pipeline.py             # Main processing pipeline
```

## Recent Optimizations (2025-12-31)

- ✅ Simplified AIP to direct template difference (-54% code)
- ✅ Removed 14 unused methods, 36 unused parameters
- ✅ Renamed `preprocessing_*` to `AIP_*` for clarity
- ✅ Reduced config.py from 225 to 107 lines (-52%)
- ✅ 100% accuracy maintained with simpler approach

## License

[Your License Here]
