# VLM PDF Recognizer

A local, privacy-first document processing system that combines computer vision alignment with Vision Language Model (VLM) inference for zero-shot content recognition on Traditional Chinese scanned documents.

> **No cloud services. No training data. No fine-tuning.**
> Drop in scanned PDFs, get structured field extraction results in seconds.

---

## Overview

VLM PDF Recognizer processes scanned PDF documents through a multi-stage pipeline: **template matching** identifies the document type, **geometric alignment** corrects perspective distortion, **ROI extraction** isolates predefined fields, **AIP preprocessing** detects content presence via pixel-level template differencing, and **VLM recognition** extracts text from populated fields using a local Ollama-hosted model.

The system is designed for batch processing of structured documents where fields occupy known locations across a small set of document templates.

### Pipeline Architecture

```
Scanned PDF / Image
       │
       ▼
┌──────────────────┐
│  PDF → BGR Image │  PyMuPDF conversion, preserving original dimensions
└────────┬─────────┘
         ▼
┌──────────────────┐
│  SIFT Feature    │  Scale-invariant keypoint detection
│  Extraction      │  Templates: unlimited features, Documents: capped at 5000
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Template Voting │  FLANN + Lowe's ratio test → RANSAC homography
│  & Classification│  Winner = template with max inliers (min 50 required)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Geometric       │  Perspective warp via homography matrix
│  Alignment       │  Output matches template pixel dimensions exactly
└────────┬─────────┘
         ▼
┌──────────────────┐
│  ROI Extraction  │  Crop predefined field regions from aligned image
└────────┬─────────┘
         ▼
┌───────────────────────────────────────────────┐
│  Two-Stage Content Detection                  │
│                                               │
│  Stage 1: AIP (Advanced Image Processing)     │
│  ┌─────────────────────────────────────────┐  │
│  │ ECC sub-pixel alignment to blank ROI    │  │
│  │ BGR channel difference computation      │  │
│  │ Multi-stage thresholding:               │  │
│  │   • Mean diff > 0.01 → content present  │  │
│  │   • Mean diff > 0.15 → pre-printed text │  │
│  │     → check significant pixel ratio     │  │
│  │ Result: has_content (bool), ~10-30ms    │  │
│  └─────────────────────────────────────────┘  │
│                                               │
│  Stage 2: VLM Recognition (Ollama glm-ocr)    │
│  ┌─────────────────────────────────────────┐  │
│  │ Field-specific prompts per ROI type     │  │
│  │ Base64 image → Ollama HTTP API          │  │
│  │ Post-processing:                        │  │
│  │   • Prompt echo removal                 │  │
│  │   • Simplified → Traditional Chinese    │  │
│  │   • JSON response parsing               │  │
│  │ Result: content_text, ~0.5-1.5s/ROI     │  │
│  └─────────────────────────────────────────┘  │
└──────────────────┬────────────────────────────┘
                   ▼
┌───────────────────────────────────────┐
│  Output: JSON + CSV + Visualization   │
│  Case-level & document-level results  │
│  Color-coded ROI bounding boxes       │
└───────────────────────────────────────┘
```

---

## Key Features

### Template-Based Document Classification
- SIFT feature extraction with FLANN nearest-neighbor matching
- RANSAC-based homography estimation with voting mechanism
- Minimum 50 inlier threshold for reliable classification
- Supports multiple document templates simultaneously

### Geometric Alignment
- Perspective correction via homography transformation
- Aligned output matches template dimensions pixel-for-pixel
- Enables precise ROI extraction regardless of scan angle or position

### AIP (Advanced Image Processing) Content Detection
- ECC sub-pixel alignment between document ROI and blank template reference
- BGR channel difference with multi-stage adaptive thresholding
- Handles pre-printed text fields (high baseline difference) separately from handwritten content
- ~10-30ms per ROI, enabling fast pre-screening before VLM inference

### VLM-Powered Content Extraction
- Ollama-hosted `glm-ocr` model for local, GPU-accelerated inference
- Zero-shot recognition: no training or fine-tuning required
- Field-type-specific prompts optimized for:
  - **Checkboxes** — presence detection (checked/unchecked)
  - **Stamps/Seals** — presence and text extraction
  - **Text fields** — Traditional Chinese character extraction
  - **Number fields** — digit extraction (dates, ID numbers, version codes)
  - **Person number** — Taiwan national ID format (1 letter + 9 digits)
- Automatic Simplified → Traditional Chinese conversion via OpenCC

### Batch Processing with Nested Directory Support
- Input structure: `input/{date}/{case_id}/*.pdf`
- Output mirrors input: `output/{date}/{case_id}/`
- Case-level aggregation: a case passes only if all its documents pass
- CSV export for downstream analysis

### Document Validation Logic
- **VX1 priority rule**: if disagreement checkbox is checked → document fails
- **Date fields (OR logic)**: at least one of year/month/date must have content
- **Required fields (AND logic)**: all non-date, non-checkbox, non-version fields must have content

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.9+ |
| GPU | NVIDIA GPU with 6+ GB VRAM (recommended) |
| Ollama | Installed and running on `localhost:11434` |
| Model | `glm-ocr` (auto-pulled on first run if not present) |
| OS | Linux / Windows / macOS |

**Tested on:** RTX 4080 Laptop (12.5 GB VRAM), Ubuntu Linux

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/VLM-pdfRecognizer.git
cd VLM-pdfRecognizer

# Create conda environment
conda create -n vlmcv python=3.9
conda activate vlmcv

# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not already installed)
# See: https://ollama.com/download

# Pull the VLM model
ollama pull glm-ocr
```

### Dependencies

```
opencv-python>=4.8.0              # Image processing & SIFT features
numpy>=1.24.0                     # Array operations
PyMuPDF>=1.23.0                   # PDF → image conversion
requests>=2.28.0                  # Ollama HTTP API client
Pillow>=10.0.0                    # Image I/O utilities
opencc-python-reimplemented       # Simplified → Traditional Chinese
```

Optional: `torch` (GPU VRAM auto-detection; falls back to `nvidia-smi` if absent)

---

## Quick Start

### 1. Generate Configuration Files (First Time)

```bash
python update_configs.py
```

This reads LabelMe annotations from `templates/location/` and generates:
- `data/{template_id}/config.json` — ROI coordinates
- `data/{template_id}/blank_rois/*.png` — blank reference images for AIP
- `vlm_pdf_recognizer/recognition/field_schema.py` — field definitions and prompts

### 2. Start Ollama Server

```bash
ollama serve
```

### 3. Run the Pipeline

```bash
# Full pipeline (AIP + VLM)
python main.py

# AIP only (skip VLM inference)
python main.py --disable-vlm
```

### 4. Check Results

```
output/
└── {date}/
    ├── VLM_results.json                         # Aggregated results for this date
    ├── vlm_recognition_results.csv              # Tabular export for this date
    └── {case_id}/
        ├── {doc}_visualization.png              # Color-coded ROI boxes
        ├── metadata/{doc}_metadata.json         # Processing metadata
        ├── rois/{doc}_roi_{field}.png            # Extracted ROI images
        └── processed_rois/{doc}_roi_{field}_processed.png  # AIP diff images
```

---

## Output Format

### VLM_results.json

```json
{
  "preprocessing": {
    "total_documents": 10,
    "successful": 10,
    "failed": 0,
    "average_processing_time_ms": 2341.5,
    "template_distribution": {
      "contractor_1": 5,
      "contractor_2": 3,
      "enterprise_1": 2
    }
  },
  "vlm_recognition": {
    "total_documents": 10,
    "successful": 8,
    "failed": 2,
    "average_processing_time_ms": 4520.3
  },
  "case_results": {
    "case_a101": {
      "case_results": true,
      "document_count": 2,
      "valid_count": 2
    }
  },
  "documents": [
    {
      "document_ID": "doc1.pdf",
      "results": true,
      "type": "contractor_1",
      "case_id": "case_a101",
      "fields": {
        "VX1": { "has_content": false },
        "person1": { "has_content": true, "content_text": "王小明" },
        "big1": { "has_content": true }
      }
    }
  ]
}
```

### Visualization Color Code

| Color | Meaning |
|-------|---------|
| Green | Content detected (`has_content: true`) |
| Red | No content (`has_content: false`) |
| Blue | Detection unavailable (`has_content: null`) |

---

## Project Structure

```
VLM-pdfRecognizer/
│
├── main.py                          # Entry point & batch orchestrator
├── update_configs.py                # Config generator (LabelMe → config.json + field_schema.py)
├── requirements.txt
│
├── templates/
│   ├── images/                      # Golden template images (.jpg)
│   └── location/                    # LabelMe annotation files (.json)
│
├── data/                            # Auto-generated by update_configs.py
│   └── {template_id}/
│       ├── config.json              # ROI coordinates & metadata
│       ├── template_features.pkl    # Cached SIFT descriptors
│       └── blank_rois/              # Blank ROI reference images
│
├── input/                           # Place documents here
│   └── {date}/{case_id}/*.pdf
│
├── output/                          # Processing results
│
└── vlm_pdf_recognizer/
    ├── pipeline.py                  # DocumentProcessor: orchestrates full pipeline
    ├── output.py                    # Result serialization, visualization, CSV
    │
    ├── preprocessing/
    │   └── pdf_converter.py         # PDF → BGR image array (PyMuPDF)
    │
    ├── alignment/
    │   ├── feature_extractor.py     # SIFT keypoint & descriptor extraction
    │   ├── template_matcher.py      # FLANN matching + RANSAC voting
    │   ├── geometric_corrector.py   # Homography-based perspective warp
    │   └── blank_template_roi_cache.py  # In-memory cache for blank ROI references
    │
    ├── extraction/
    │   └── roi_extractor.py         # ROI cropping & bounding box drawing
    │
    ├── recognition/
    │   ├── vlm_loader.py            # Ollama client (singleton, auto GPU detection)
    │   ├── vlm_recognizer.py        # VLM inference, post-processing, validation
    │   ├── roi_preprocessor.py      # AIP: ECC alignment + BGR differencing
    │   ├── field_schema.py          # Field definitions & prompt templates (auto-generated)
    │   └── csv_exporter.py          # Flat CSV export
    │
    └── templates/
        ├── __init__.py              # ROI & GoldenTemplate dataclasses
        ├── template_loader.py       # Load templates, configs, features
        └── template_cache.py        # SIFT feature serialization & cache invalidation
```

### Module Responsibilities

| Module | Role |
|--------|------|
| `pipeline.py` | Orchestrates preprocessing: PDF conversion → SIFT → template match → alignment → ROI extraction |
| `vlm_recognizer.py` | Runs AIP + VLM on extracted ROIs, aggregates per-document results, applies validation rules |
| `roi_preprocessor.py` | AIP engine: ECC-aligned BGR differencing against blank template ROIs |
| `vlm_loader.py` | Singleton Ollama HTTP client with hardware auto-detection and model availability checks |
| `field_schema.py` | Central registry of field metadata, types, and VLM prompt templates (auto-generated) |
| `template_matcher.py` | FLANN + RANSAC voting to classify documents and compute homography matrices |
| `output.py` | Saves visualizations, metadata JSON, ROI images, and batch summary |
| `csv_exporter.py` | Flattens recognition results into a single CSV with case-level aggregation |
| `update_configs.py` | One-time generator: LabelMe annotations → config.json + blank_rois + field_schema.py |

---

## Configuration & Tuning

### AIP Thresholds (`roi_preprocessor.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_ABSOLUTE_DENSITY_THRESHOLD` | 0.01 | Mean BGR difference above which a field is considered to have content |
| `significant_threshold` | 30 | Per-pixel difference must exceed this to count as significant |
| `mean_diff > 0.15` | — | Triggers pre-printed text handling (higher baseline) |
| `significant_ratio > 0.20` | — | For pre-printed fields, ratio of significant pixels required |

### Template Matching (`template_matcher.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ratio_threshold` | 0.7 | Lowe's ratio test for FLANN matches |
| `ransac_threshold` | 5.0 | RANSAC reprojection error tolerance (pixels) |
| `min_inlier_count` | 50 | Minimum RANSAC inliers to accept a template match |

### VLM Inference (`vlm_loader.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `glm-ocr` | Ollama model identifier |
| `temperature` | 0.0 | Deterministic output |
| `num_predict` | 256 | Maximum generated tokens |
| `API timeout` | 120s | Per-request timeout |

---

## Performance

Measured on RTX 4080 Laptop (12.5 GB VRAM):

| Stage | Time |
|-------|------|
| PDF conversion | < 100 ms |
| SIFT extraction | 100–300 ms |
| Template matching (FLANN + RANSAC) | 200–500 ms |
| Geometric alignment | 50–100 ms |
| ROI extraction | < 100 ms |
| AIP preprocessing (per ROI) | 10–30 ms |
| VLM inference (per ROI) | 0.5–1.5 s |
| **Total per document** | **3–5 s** |

---

## Adding New Templates

1. Create a template image and save to `templates/images/{template_id}.jpg`
2. Annotate ROI regions using [LabelMe](https://github.com/wkentaro/labelme) and save to `templates/location/{template_id}.json`
3. Add the template ID to `load_all_templates()` in `template_loader.py`
4. Run `python update_configs.py` to generate configs, blank ROIs, and update field schemas
5. Verify with `python main.py`

---

## Security Note on Input/Output Data

The `input/` and `output/` directories are excluded from version control (via `.gitignore`) as they contain confidential documents. However, the system is designed with strong generalizability — simply follow the setup steps above and place your own PDF documents in the `input/{date}/{case_id}/` directory structure. The pipeline works well with a wide variety of structured document types, including authorization letters, consent forms, contracts, and other similar documents.

---

## License

This project is available for educational and portfolio demonstration purposes.

## Contact

For questions, issues, or collaboration inquiries:

- Developer: Tom Huang
- Email: huang1473690@gmail.com
