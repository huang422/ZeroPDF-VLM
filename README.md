# VLM PDF Recognizer

Local, zero-shot document processing system for Traditional Chinese scanned PDF documents with template-based classification, alignment, and ROI extraction.

## Features

- **Template Classification**: Automatically identifies document type (enterprise_1, contractor_1, contractor_2)
- **Watermark Removal**: HSV thresholding to remove blue, gray, and light red watermarks
- **Geometric Alignment**: SIFT feature matching with homography transformation
- **ROI Extraction**: Extracts predefined regions of interest with bounding box visualization
- **Batch Processing**: Process entire directories with error logging
- **GPU Acceleration**: Automatic GPU detection with CPU fallback

## Requirements

- Python 3.9+
- OpenCV 4.x
- 4GB VRAM (GPU mode) or 8GB RAM (CPU mode)

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

```bash
python main.py
```

That's it! The program will:
- Load all templates from `templates/images/`
- Process all files in `input/` directory
- Save results to `output/` directory

## Output Files

For each processed document, you'll get:

```
output/
├── 101_aligned.png              # Geometrically aligned document
├── 101_visualization.png        # Aligned + ROI bounding boxes
├── 101_metadata.json            # Processing metadata
├── 101_roi_enterprise_1_title.png   # Individual ROI extractions
├── 101_roi_VX1.png
├── ...
└── batch_summary.json           # Overall batch statistics
```

### Metadata Example

```json
{
  "input_path": "input/101.jpg",
  "matched_template_id": "enterprise_1",
  "confidence_score": 156,
  "processing_time_ms": 2341.5,
  "success": true,
  "roi_count": 17,
  "rois": [...]
}
```

## Performance

- **CPU Mode**: ~2-3 seconds per document
- **Feature Caching**: First run computes SIFT features, subsequent runs are 40% faster

## License

[Your License Here]
