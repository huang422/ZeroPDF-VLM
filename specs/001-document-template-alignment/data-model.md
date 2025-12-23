# Data Model: Document Template Alignment & ROI Extraction

**Feature**: 001-document-template-alignment
**Date**: 2025-12-23
**Phase**: 1 - Design

## Overview

This document defines the data structures and entities used in the document template alignment system. All models are designed for file-based storage and processing.

## Core Entities

### 1. GoldenTemplate

Represents a reference template for document classification and alignment.

**Fields**:
- `template_id`: str - Unique identifier (e.g., "enterprise_1", "contractor_1", "contractor_2")
- `template_image_path`: str - Path to template image file (PNG format)
- `config_path`: str - Path to JSON configuration file
- `features_cache_path`: str - Path to cached SIFT features (PKL file)
- `keypoints`: List[cv2.KeyPoint] - SIFT keypoints (loaded from cache or computed)
- `descriptors`: np.ndarray - SIFT descriptors (NxM matrix)
- `image_shape`: Tuple[int, int, int] - (height, width, channels) of template image
- `rois`: List[ROI] - Regions of interest defined for this template

**Validation Rules**:
- `template_id` must match one of: ["enterprise_1", "contractor_1", "contractor_2"]
- `template_image_path` must exist and be readable
- `config_path` must exist and contain valid JSON
- If `features_cache_path` exists and is newer than template image, load from cache
- Otherwise, compute features and save to cache

**State Transitions**:
- Unloaded → Loading → Loaded (features cached)
- Loaded → Stale (if template image modified) → Re-computing → Loaded

**File Location**: `data/{template_id}/`

---

### 2. ROI (Region of Interest)

Represents a specific rectangular area on a template for data extraction.

**Fields**:
- `roi_id`: str - Unique identifier within template (e.g., "company_name", "date")
- `description`: str - Human-readable description
- `x1`: int - Top-left X coordinate
- `y1`: int - Top-left Y coordinate
- `x2`: int - Bottom-right X coordinate
- `y2`: int - Bottom-right Y coordinate
- `format`: str - Always "top_left_bottom_right"

**Validation Rules**:
- `x1 < x2` and `y1 < y2` (valid rectangle)
- All coordinates must be non-negative
- Coordinates must be within template image bounds
- `roi_id` must be unique within template

**Relationships**:
- Belongs to exactly one GoldenTemplate
- Multiple ROIs per template allowed

**Serialization**: JSON format in template config file

---

### 3. InputDocument

Represents a document to be processed (PDF or image).

**Fields**:
- `file_path`: str - Path to input file
- `file_type`: str - One of: ["pdf", "png", "jpg", "jpeg"]
- `pages`: List[DocumentPage] - List of pages (1 for images, N for PDFs)
- `total_pages`: int - Number of pages

**Validation Rules**:
- `file_path` must exist and be readable
- `file_type` must match actual file extension
- For PDFs: `total_pages >= 1`
- For images: `total_pages == 1`

**State Transitions**:
- Created → Loaded → Pages Extracted → Processing → Completed/Failed

---

### 4. DocumentPage

Represents a single page from an input document (extracted from PDF or direct image).

**Fields**:
- `page_number`: int - Page index (0-based)
- `image_array`: np.ndarray - BGR image array (HxWx3)
- `original_dimensions`: Tuple[int, int] - (height, width) before any processing
- `preprocessed_image`: Optional[np.ndarray] - After watermark removal and binarization
- `keypoints`: Optional[List[cv2.KeyPoint]] - SIFT keypoints (computed during matching)
- `descriptors`: Optional[np.ndarray] - SIFT descriptors
- `matched_template`: Optional[str] - ID of matched template
- `match_confidence`: Optional[float] - Ratio of inliers to total matches
- `inlier_count`: Optional[int] - Number of inlier matches
- `homography_matrix`: Optional[np.ndarray] - 3x3 transformation matrix
- `aligned_image`: Optional[np.ndarray] - Image after perspective warp
- `processing_status`: str - One of: ["pending", "preprocessing", "matching", "aligning", "extracting", "completed", "failed"]
- `error_message`: Optional[str] - Error details if status == "failed"

**Validation Rules**:
- `image_array` must be 3-channel BGR format
- `match_confidence` range: [0.0, 1.0]
- `inlier_count >= 50` for successful match (threshold from FR-016)
- `homography_matrix` must be 3x3 if present

**State Transitions**:
```
pending → preprocessing → matching → aligning → extracting → completed
                                                            ↘ failed
```

---

### 5. FeatureMatch

Represents a match between input document and template features.

**Fields**:
- `template_id`: str - ID of template being matched against
- `total_matches`: int - Total feature matches found
- `good_matches`: List[cv2.DMatch] - Matches passing Lowe's ratio test
- `inliers`: np.ndarray - Boolean mask from RANSAC (1 = inlier, 0 = outlier)
- `inlier_count`: int - Number of inliers
- `confidence`: float - inlier_count / total_matches

**Validation Rules**:
- `inlier_count <= total_matches`
- `confidence` range: [0.0, 1.0]
- `good_matches` filtered with ratio < 0.7 (Lowe's ratio test)

**Relationships**:
- One FeatureMatch per (DocumentPage, GoldenTemplate) pair
- DocumentPage has 3 FeatureMatches (one per template)
- Winner selected based on highest `inlier_count` (voting mechanism from FR-009)

---

### 6. ProcessingResult

Represents the output of processing a single document page.

**Fields**:
- `input_file_path`: str - Original input file
- `page_number`: int - Page index
- `matched_template_id`: str - ID of matched template
- `match_confidence`: float - Confidence score [0.0, 1.0]
- `inlier_count`: int - Number of inlier feature matches
- `alignment_success`: bool - Whether alignment succeeded
- `extracted_rois`: List[ExtractedROI] - ROI regions extracted from aligned image
- `output_image_path`: str - Path to saved verification image with ROI boxes
- `processing_time_seconds`: float - Total processing time
- `error_message`: Optional[str] - Error if processing failed

**Validation Rules**:
- `match_confidence >= 0.5` recommended (not enforced, logged as warning)
- `inlier_count >= 50` required (from FR-016)
- `alignment_success == True` required for ROI extraction
- `output_image_path` must be written to output/ directory

---

### 7. ExtractedROI

Represents a cropped ROI region from aligned document.

**Fields**:
- `roi_id`: str - ID from template configuration
- `description`: str - Human-readable description
- `bounding_box`: Tuple[int, int, int, int] - (x1, y1, x2, y2) in aligned image coordinates
- `roi_image`: np.ndarray - Cropped image region
- `visualization_color`: Tuple[int, int, int] - BGR color for bounding box overlay (default: green (0, 255, 0))

**Validation Rules**:
- `bounding_box` must be within aligned image bounds
- `roi_image` dimensions must match (x2-x1, y2-y1)

**Relationships**:
- Belongs to one ProcessingResult
- Corresponds to one ROI definition from GoldenTemplate

---

### 8. BatchProcessingJob

Represents a batch processing operation on multiple documents.

**Fields**:
- `job_id`: str - Unique identifier (UUID or timestamp-based)
- `input_directory`: str - Directory containing input documents
- `output_directory`: str - Directory for outputs
- `input_files`: List[str] - Paths to all input files
- `total_files`: int - Number of files to process
- `completed_count`: int - Number successfully processed
- `failed_count`: int - Number failed
- `processing_results`: List[ProcessingResult] - Results for each processed page
- `failed_files`: List[Tuple[str, str]] - (file_path, error_message) for failures
- `start_time`: datetime - Job start timestamp
- `end_time`: Optional[datetime] - Job completion timestamp
- `status`: str - One of: ["queued", "running", "completed", "partial_failure"]

**Validation Rules**:
- `completed_count + failed_count <= total_files`
- `status == "completed"` only if `completed_count == total_files`
- `status == "partial_failure"` if `failed_count > 0` and `completed_count > 0`

**State Transitions**:
```
queued → running → completed (if all succeed)
                 → partial_failure (if some fail, continue per FR-024)
```

---

## Data Flow

```
InputDocument (PDF/Image)
    ↓
DocumentPage(s) extracted with dimension preservation
    ↓
Preprocessing: watermark removal → binary image
    ↓
Feature Extraction: SIFT keypoints & descriptors
    ↓
Feature Matching: against all 3 GoldenTemplates → 3 FeatureMatch objects
    ↓
Template Selection: max(inlier_count) → matched_template_id
    ↓
Homography Computation: RANSAC → homography_matrix
    ↓
Geometric Alignment: warpPerspective → aligned_image
    ↓
ROI Extraction: crop regions → ExtractedROI objects
    ↓
Visualization: draw bounding boxes → output_image
    ↓
ProcessingResult saved to output/ directory
```

## File Structure Mapping

```
data/
├── enterprise_1/
│   ├── template.png              → GoldenTemplate.template_image_path
│   ├── template_features.pkl     → GoldenTemplate.features_cache_path
│   └── config.json               → GoldenTemplate.config_path (contains ROI definitions)
├── contractor_1/
│   └── [same structure]
└── contractor_2/
    └── [same structure]

output/
└── {timestamp}_{filename}_page{N}/
    ├── aligned_with_rois.png     → ProcessingResult.output_image_path
    └── metadata.json             → ProcessingResult serialized
```

## Configuration File Schema

### Template Config (config.json)

```json
{
  "template_name": "enterprise_1",
  "template_version": "1.0",
  "image_dimensions": {
    "width": 2480,
    "height": 3508,
    "unit": "pixels"
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
      },
      "format": "top_left_bottom_right"
    }
  ]
}
```

### Feature Cache (template_features.pkl)

```python
{
  'keypoints': [(pt, size, angle, response, octave, class_id), ...],
  'descriptors': np.ndarray,  # NxM SIFT descriptors
  'image_shape': (height, width, channels),
  'template_name': str,
  'created_at': ISO 8601 timestamp
}
```

## Error Handling

**Error Types**:
1. **UnknownDocumentError**: Raised when `inlier_count < 50` for all templates (FR-016)
2. **InvalidTemplateError**: Template image or config missing/corrupted
3. **PDFConversionError**: Failed to convert PDF to images
4. **AlignmentError**: Homography computation failed or produced invalid matrix
5. **ROIExtractionError**: ROI coordinates out of bounds after alignment

**Error Recovery** (Batch Mode - FR-024):
- Log error with file path and error type
- Continue processing remaining files
- Report all failures in final summary

## Performance Considerations

1. **Feature Cache**: Load cached SIFT features for templates on startup (avoid recomputation per FR-015)
2. **Memory**: Process one page at a time, release after writing output
3. **GPU Acceleration**: Check `cv2.cuda.getCudaEnabledDeviceCount()` on startup, use if available (FR-017)
4. **Batch Processing**: Process sequentially but fail independently (FR-024)

## Summary

- **8 core entities** defined with clear validation rules
- **File-based storage** for templates and outputs
- **State machines** for document processing workflow
- **Error types** mapped to functional requirements
- **Performance optimizations** via caching and GPU detection
