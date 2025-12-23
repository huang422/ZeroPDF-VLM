# Research: Document Template Alignment & ROI Extraction

**Feature**: 001-document-template-alignment
**Date**: 2025-12-23
**Phase**: 0 - Technical Research

## Overview

This document captures research findings for technical decisions required to implement the document template alignment system.

## 1. PDF to Image Conversion Library

### Decision

**PyMuPDF (fitz)** - Recommended choice

### Rationale

1. **Performance**: Fastest among Python PDF libraries, written in C with Python bindings
2. **Dimension Preservation**: Provides exact control over output resolution and dimensions via matrix parameter
3. **Dependencies**: Pure Python wheel available (no external dependencies like poppler required)
4. **Memory Efficiency**: Loads pages individually, avoiding full document in memory
5. **Licensing**: AGPL (free for open source) with commercial licensing available
6. **Integration**: Returns PIL Image or numpy array directly, seamless OpenCV integration

### Alternatives Considered

**pdf2image**:
- Pros: Simple API, widely used
- Cons: Requires poppler-utils system dependency, slower performance, less control over output dimensions
- Rejected because: External dependency complicates deployment, slower for batch processing

**pypdfium2**:
- Pros: Apache 2.0 license (more permissive), no external dependencies
- Cons: Less mature, smaller community, limited documentation
- Rejected because: Less proven in production, PyMuPDF has better performance benchmarks

### Implementation Details

```python
import fitz  # PyMuPDF
import numpy as np
import cv2

def pdf_to_images(pdf_path):
    """
    Convert multi-page PDF to list of numpy arrays preserving original dimensions.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of numpy arrays (BGR format for OpenCV)

    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If PDF is corrupted
    """
    try:
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get page at original resolution (matrix=fitz.Identity preserves dimensions)
            # For higher quality: use matrix=fitz.Matrix(2, 2) for 2x scaling
            pix = page.get_pixmap(matrix=fitz.Identity)

            # Convert to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # Convert RGB to BGR for OpenCV
            if pix.n == 3:  # RGB
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:  # RGBA
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:  # Grayscale
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            images.append(img_bgr)

        doc.close()
        return images

    except Exception as e:
        raise ValueError(f"Failed to convert PDF: {str(e)}")
```

### Installation Requirements

```bash
pip install PyMuPDF
```

Note: PyMuPDF provides pre-built wheels for Linux, no compilation needed.

## 2. SIFT Feature Extraction Best Practices

### Decision

Use **cv2.SIFT_create()** with optimized parameters for document matching

### Rationale

1. **Patent Status**: SIFT patent expired in 2020, now available in opencv-python (not just opencv-contrib-python)
2. **Robustness**: Scale and rotation invariant, ideal for skewed scanned documents
3. **Accuracy**: Superior to ORB for document alignment tasks (more keypoints, better descriptors)
4. **Performance**: Hardware acceleration available on GPU via OpenCV CUDA build

### Recommended Parameters

```python
# For document templates (high quality, clean scans)
sift = cv2.SIFT_create(
    nfeatures=0,          # No limit on features (extract all)
    nOctaveLayers=3,      # Standard pyramid layers
    contrastThreshold=0.04,  # Standard threshold
    edgeThreshold=10,     # Standard edge rejection
    sigma=1.6             # Standard Gaussian blur
)

# For input documents (may have noise, watermarks)
sift_input = cv2.SIFT_create(
    nfeatures=5000,       # Limit to top 5000 features (performance)
    nOctaveLayers=3,
    contrastThreshold=0.03,  # Lower threshold to catch more features
    edgeThreshold=10,
    sigma=1.6
)
```

### Feature Matching Strategy

**FLANN-based matcher** for speed:

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Balance between speed and accuracy
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Use KNN matching with Lowe's ratio test
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
```

Alternative: **BFMatcher** for accuracy (slower):
```python
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
```

## 3. Watermark Removal - HSV Thresholding Parameters

### Decision

Multi-stage thresholding targeting specific watermark colors

### Rationale

Scanned documents typically have:
- Blue watermarks (low S, high V in HSV)
- Light gray/red watermarks (low S, high V)
- Black text/stamps (low V in HSV)
- Preserve: Black text, dark blue pen, red stamps

### Recommended Approach

```python
def remove_watermarks(image):
    """
    Remove blue, light gray, light red watermarks while preserving content.

    Args:
        image: BGR image from OpenCV

    Returns:
        Binary image (0 or 255 only)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Create mask for dark content (black text, stamps, pen marks)
    # Keep pixels with low Value (dark colors)
    dark_mask = cv2.inRange(v, 0, 180)  # Adjust upper bound based on testing

    # Alternative: Use saturation to preserve red stamps
    # Red has high saturation even when bright
    red_mask = cv2.inRange(s, 100, 255)

    # Combine masks
    content_mask = cv2.bitwise_or(dark_mask, red_mask)

    # Apply morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)

    # Create binary output
    binary_output = cv2.threshold(content_mask, 127, 255, cv2.THRESH_BINARY)[1]

    return binary_output
```

**Note**: Parameters (especially V threshold = 180) should be tuned on actual document samples.

## 4. Homography Computation - RANSAC Parameters

### Decision

Use **cv2.findHomography()** with RANSAC and optimized parameters

### Rationale

RANSAC is essential to reject outliers from:
- Handwritten annotations
- Stamps
- Content differences between template and input

### Recommended Parameters

```python
# Minimum matches required before attempting homography
MIN_MATCH_COUNT = 50

if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography with RANSAC
    M, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransacReprojThreshold=5.0  # Pixel tolerance for inliers (tune based on image resolution)
    )

    # Count inliers for confidence metric
    inliers = mask.ravel().sum()
    confidence = inliers / len(good_matches)

else:
    raise ValueError(f"Insufficient matches: {len(good_matches)} (need {MIN_MATCH_COUNT})")
```

### Performance Optimization

For GPU acceleration (if cv2.cuda available):
```python
# Check CUDA availability
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Use GPU-accelerated SIFT if available
    # Note: Requires opencv-contrib-python compiled with CUDA
    pass
else:
    # Fallback to CPU
    pass
```

## 5. ROI Configuration Format

### Decision

JSON format with explicit coordinate specification

### Structure

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
    },
    {
      "id": "date",
      "description": "Document date",
      "coordinates": {
        "x1": 1500,
        "y1": 150,
        "x2": 1900,
        "y2": 250
      },
      "format": "top_left_bottom_right"
    }
  ]
}
```

### Rationale

- Explicit coordinate format prevents ambiguity
- Image dimensions allow validation
- ROI IDs enable programmatic access
- Descriptions aid manual verification

## 6. Feature Cache Strategy

### Decision

Pickle serialization for SIFT features and descriptors

### Format

```python
import pickle

# Save template features
cache_data = {
    'keypoints': [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                  for kp in keypoints],
    'descriptors': descriptors,
    'image_shape': template_image.shape,
    'template_name': 'enterprise_1',
    'created_at': datetime.now().isoformat()
}

with open('data/enterprise_1/template_features.pkl', 'wb') as f:
    pickle.dump(cache_data, f)

# Load template features
def load_cached_features(cache_path):
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    # Reconstruct cv2.KeyPoint objects
    keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=pt[1],
                               angle=pt[2], response=pt[3],
                               octave=pt[4], class_id=pt[5])
                 for pt in data['keypoints']]

    return keypoints, data['descriptors']
```

### Validation

Cache invalidation based on:
- Template image modification time
- OpenCV version changes
- Explicit user request

## Summary

All technical decisions resolved:

1. ✅ **PDF Library**: PyMuPDF (fitz) - fast, no external deps, dimension control
2. ✅ **SIFT Parameters**: Optimized for document matching with FLANN matcher
3. ✅ **Watermark Removal**: HSV V-channel thresholding with adjustable parameters
4. ✅ **Homography**: RANSAC with 5.0 pixel threshold, 50 minimum matches
5. ✅ **ROI Format**: JSON with explicit coordinates and validation metadata
6. ✅ **Feature Cache**: Pickle serialization with reconstruction logic

Ready for Phase 1: Design & Contracts.
