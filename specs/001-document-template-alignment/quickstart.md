# Quickstart Guide: Document Template Alignment & ROI Extraction

**Feature**: 001-document-template-alignment
**Date**: 2025-12-23

## Prerequisites

- Python 3.9+ (vlmcv environment)
- Linux system (tested on Ubuntu/Debian)
- Optional: CUDA-enabled GPU with 4GB+ VRAM

## Installation

```bash
# Activate vlmcv environment
conda activate vlmcv  # or source vlmcv/bin/activate

# Install dependencies
pip install opencv-python numpy PyMuPDF
```

## Setup Golden Templates

Create the following directory structure:

```
data/
├── enterprise_1/
│   ├── template.png
│   └── config.json
├── contractor_1/
│   ├── template.png
│   └── config.json
└── contractor_2/
    ├── template.png
    └── config.json
```

Each `config.json` follows this format:

```json
{
  "template_name": "enterprise_1",
  "template_version": "1.0",
  "image_dimensions": {
    "width": 2480,
    "height": 3508
  },
  "rois": [
    {
      "id": "company_name",
      "description": "Company name field",
      "coordinates": {
        "x1": 200,
        "y1": 150,
        "x2": 800,
        "y2": 250
      }
    }
  ]
}
```

## Basic Usage

### Process Single Document

```bash
python -m vlm_pdf_recognizer.cli process input.pdf --output output/
```

### Process Directory (Batch Mode)

```bash
python -m vlm_pdf_recognizer.cli batch input_folder/ --output output/
```

## Output Structure

```
output/
└── 20251223_153045_document_page0/
    ├── aligned_with_rois.png    # Aligned document with ROI boxes
    └── metadata.json             # Processing details
```

## Verification

After processing, check:

1. **Aligned Image**: Visual inspection of alignment quality
2. **ROI Boxes**: Green bounding boxes should cover correct regions
3. **Metadata**: Contains template_id, confidence score, processing time

## Performance

- **CPU Mode**: ~10 seconds per page
- **GPU Mode**: ~3 seconds per page (if CUDA available)
- **Feature Cache**: First run computes SIFT features for templates, subsequent runs use cache (40% faster)

## Troubleshooting

### "Unknown Document" Error

Document failed template matching (< 50 feature matches).

**Solutions**:
- Check document quality (scan resolution >= 150 DPI)
- Verify document type matches one of the three templates
- Inspect watermark removal results

### Low Confidence Score

Template matched but with low confidence (< 0.5).

**Solutions**:
- Check document skew (should be < 15 degrees)
- Verify document isn't severely cropped
- Review preprocessing (watermark removal may be too aggressive)

### ROI Boxes Misaligned

Boxes don't cover correct regions.

**Solutions**:
- Check template image matches actual blank form
- Verify ROI coordinates in config.json
- Review alignment quality (homography matrix may be incorrect)

## Next Steps

After verifying alignment and ROI extraction:

1. Integrate OCR for text extraction from ROI regions (out of scope for this phase)
2. Add custom templates by creating new data/{template_id}/ directories
3. Tune HSV thresholding parameters if watermark removal is insufficient
