# Research Report: ROI Preprocessing Pipeline Technical Decisions

**Feature**: 004-vlm-roi-preprocessing
**Created**: 2025-12-31
**Purpose**: Document technical decisions and rationale for preprocessing pipeline implementation

## Overview

This document consolidates research findings on image preprocessing techniques for ROI content detection in form processing. All decisions are based on computer vision best practices and optimized for the project's specific constraints (batch document processing, 300 DPI scans, CPU-based OpenCV).

---

## 1. Template Difference Approach

### Decision
Use absolute difference with binary thresholding (threshold=25) combined with CLAHE (Contrast Limited Adaptive Histogram Equalization) for illumination normalization.

### Rationale
- Template difference is a fundamental technique in document analysis that effectively isolates added content by subtracting the blank template from filled documents
- Threshold=25 (on 0-255 scale) balances detection of faint content while filtering scanner noise and JPEG artifacts
- CLAHE preprocessing normalizes illumination variations between template and document scans, improving robustness across different scanning conditions

### Alternatives Considered

| Alternative | Pros | Cons | Rejection Reason |
|------------|------|------|------------------|
| Ratio-based detection | Illumination-invariant | Prone to division-by-zero errors, complex thresholding | More complex with marginal benefit for controlled scanning |
| Adaptive thresholding | Automatically adjusts per document | Inconsistent across batch, harder to debug | Lack of determinism conflicts with explainability requirement |
| Histogram matching | Perfect illumination alignment | Computationally expensive, can distort content | Performance overhead not justified for typical office scans |

### Implementation Parameters
```python
# config.py additions
DIFFERENCE_THRESHOLD = 25  # Binary threshold (range 10-50)
USE_CLAHE_NORMALIZATION = True  # Enable illumination normalization
CLAHE_CLIP_LIMIT = 2.0  # Contrast limiting parameter
CLAHE_TILE_GRID_SIZE = (8, 8)  # Local equalization grid
MEDIAN_FILTER_SIZE = 3  # Pre-filtering for noise reduction
```

### References
- Existing codebase: `roi_comparator.py` (grayscale conversion, binary thresholding patterns)
- OpenCV CLAHE documentation: Normalizes local contrast without over-amplifying noise

---

## 2. HSV Saturation Channel for Faint Stamp Detection

### Decision
Extract HSV saturation channel and apply 2x boost factor to preserve faint colored stamps that would be lost in pure grayscale conversion.

### Rationale
- **Illumination Invariance**: Saturation represents color "purity" independent of brightness (Value channel), making it robust to scanner lighting variations
- **Faint Stamp Detection**: Faint red/blue stamps have LOW saturation (washed out) but still higher than pure grayscale form elements (black lines/text have ZERO saturation)
- **Paper Background Separation**: Paper background has VERY LOW saturation (~0-4%), creating clear separation from even faint stamps (~6-20% saturation)

### Quantitative Justification
Example faint red stamp analysis:
- RGB values: (200, 150, 150) - visually very faint
- Grayscale conversion: 0.299×200 + 0.587×150 + 0.114×150 ≈ 170
- Paper background RGB: (240, 240, 240) → Grayscale ≈ 240
- **Grayscale difference**: 240 - 170 = 70 (may be below DIFFERENCE_THRESHOLD=25 after template alignment errors)

- HSV Saturation: Stamp S ≈ 20% (51/255), Paper S ≈ 0% (0/255)
- **Saturation difference**: 51 - 0 = 51 (clear detection even after 2x boost)

### Alternatives Considered

| Alternative | Pros | Cons | Rejection Reason |
|------------|------|------|------------------|
| Lab color space (a/b channels) | Perceptually uniform, color-specific detection | More complex, requires per-color calibration | Over-engineered for general stamp detection |
| YCrCb (Cr/Cb channels) | Faster than Lab, good for red/blue separation | Less robust than HSV for general case | Similar complexity to HSV without significant benefit |
| Pure grayscale | Simplest implementation | Misses faint colored stamps | Fails primary requirement (detect low-saturation stamps) |

### Implementation Parameters
```python
# config.py additions
SATURATION_MIN_THRESHOLD = 15  # Minimum saturation to detect (6% on 0-100 scale)
SATURATION_BOOST_FACTOR = 2.0  # Amplification to make faint stamps more visible
SATURATION_VALUE_BLEND_RATIO = 0.7  # 70% saturation + 30% inverted value
```

### Implementation Strategy (Step 1)
```python
def extract_saturation_channel(roi_image: np.ndarray) -> np.ndarray:
    """Extract and enhance saturation channel."""
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Boost saturation (faint stamps: 15 → 30, background: 5 → 10)
    s_boosted = np.clip(s * SATURATION_BOOST_FACTOR, 0, 255).astype(np.uint8)

    # Blend with inverted value (captures dark content like black ink)
    v_inverted = 255 - v
    combined = cv2.addWeighted(s_boosted, 0.7, v_inverted, 0.3, 0)

    return combined
```

---

## 3. Morphological Opening for Form Line Removal

### Decision
Apply morphological opening (erosion + dilation) with rectangular kernels sized at 30% of ROI dimensions to remove horizontal and vertical form lines while preserving signatures and stamps.

### Rationale
- **Opening Operation Theory**: Erosion removes thin structures (lines), then dilation restores remaining structures (signatures) to original size
- **Directional Selectivity**: Rectangular kernels (width=30% ROI_width, height=3) only remove structures aligned with kernel direction
- **Signature Preservation**: Signatures have irregular, multi-directional strokes that survive both horizontal and vertical opening
- **30% Ratio Justification**: Form lines typically span 50-80% of ROI width; 30% kernel removes long segments while preserving short signature strokes

### Why Opening is Superior

| Operation | Effect | Use Case |
|-----------|--------|----------|
| **Opening (Erosion → Dilation)** | Removes small structures, restores large | **Line removal (our choice)** |
| Closing (Dilation → Erosion) | Fills gaps, joins nearby structures | Hole filling (opposite need) |
| Erosion only | Shrinks everything | Information loss (removes signatures too) |
| Dilation only | Expands everything | Makes problem worse (thickens lines) |
| Top-Hat (Original - Opening) | Extracts small structures | Would extract lines, not remove them |
| Black-Hat (Closing - Original) | Extracts dark holes | Not suitable for line removal |

### Alternatives Considered

| Alternative | Pros | Cons | Rejection Reason |
|------------|------|------|------------------|
| Hough Line Transform | Detects specific line angles | Computationally expensive, requires parameter tuning | Overkill for simple horizontal/vertical lines |
| Canny + Line Removal | Precise edge detection | Complex multi-step process, fragile to noise | Over-engineered for form lines |
| Larger kernel (50% ROI) | Removes more line fragments | Also removes long signature strokes | Too aggressive (false positives on signatures) |
| Smaller kernel (10% ROI) | Safer for signatures | Leaves too many line fragments | Insufficient line removal (false negatives) |

### Implementation Parameters
```python
# config.py additions
MORPHOLOGY_LINE_RATIO = 0.3  # Kernel size as ratio of ROI dimension
MIN_SIGNATURE_AREA = 500  # Minimum area to protect from line removal (pixels)
ADAPTIVE_KERNEL_SIZING = True  # Adjust ratio based on ROI size

# Adaptive sizing rules
# Large ROIs (>500px): ratio=0.4 (more aggressive)
# Medium ROIs (200-500px): ratio=0.3 (default)
# Small ROIs (<200px): ratio=0.2 (conservative)
```

### Implementation Strategy (Step 3)
```python
def remove_form_lines(binary_image: np.ndarray) -> np.ndarray:
    """Remove horizontal and vertical lines using morphological opening."""
    h, w = binary_image.shape

    # Horizontal line removal
    h_kernel_width = int(w * MORPHOLOGY_LINE_RATIO)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 3))
    h_removed = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, h_kernel)

    # Vertical line removal
    v_kernel_height = int(h * MORPHOLOGY_LINE_RATIO)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, v_kernel_height))
    v_removed = cv2.morphologyEx(h_removed, cv2.MORPH_OPEN, v_kernel)

    return v_removed
```

---

## 4. Connected Components Analysis for Content Detection

### Decision
Use 8-connectivity connected components analysis with multi-metric filtering: minimum blob area (20 pixels), component count threshold (≥3), aspect ratio limit (<10), and combined ink ratio decision logic.

### Rationale
- **Component Count Differentiates Content Type**:
  - Form residue: Many small fragments (10-100+ tiny blobs)
  - Signature: Fewer larger coherent blobs (3-15 stroke segments)
  - Stamp: Moderate count (5-50 character/pattern blobs)
- **Area Threshold Filters Noise**: 20 pixels minimum removes salt-and-pepper artifacts while preserving signature stroke segments
- **Aspect Ratio Filters Line Fragments**: Form line residue has extreme ratios (>10:1); signatures have moderate ratios (1:1 to 5:1)
- **8-Connectivity**: Connects diagonal pixels, better for detecting coherent signature strokes than 4-connectivity

### Quantitative Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| MIN_BLOB_AREA | 20 pixels | Adaptive: 0.1% of ROI area, min 20 px |
| COMPONENT_COUNT_THRESHOLD | 3 | Signatures: 3-15 components, residue: 10-100+ |
| MAX_ASPECT_RATIO | 10 | Signatures: 1-5, line fragments: >10 |
| MIN_DENSITY | 0.3 | Signatures: 30-80% dense, noise: <30% |
| INK_RATIO_THRESHOLD | 0.005 (0.5%) | Combined metric fallback |

### Alternatives Considered

| Alternative | Pros | Cons | Rejection Reason |
|------------|------|------|------------------|
| Ink ratio only | Simple single metric | Misses sparse signatures | Insufficient discrimination |
| Contour analysis | More geometric features | Slower, requires tuning | Over-complex for binary decision |
| HOG features | Detects oriented gradients | Computationally expensive, requires training | ML approach conflicts with rule-based requirement |
| Template matching | Learns signature patterns | Requires labeled data, not generalizable | Out of scope (explainability, no training data) |

### Implementation Parameters
```python
# config.py additions
MIN_BLOB_AREA = 20  # Minimum component area in pixels
COMPONENT_COUNT_THRESHOLD = 3  # Minimum valid components for content
MAX_ASPECT_RATIO = 10  # Maximum width/height ratio for valid component
MIN_COMPONENT_DENSITY = 0.3  # Minimum (area / bounding_box_area)
INK_RATIO_THRESHOLD = 0.005  # Minimum total ink ratio (0.5%)
USE_8_CONNECTIVITY = True  # Connect diagonal pixels
```

### Implementation Strategy (Steps 4-6)
```python
def analyze_connected_components(binary_image: np.ndarray) -> tuple[int, float]:
    """Extract features: valid component count and ink ratio."""
    h, w = binary_image.shape

    # Extract components (8-connectivity)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    # Filter components
    valid_count = 0
    total_valid_area = 0

    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, width, height, area = stats[i]

        # Filter by area
        if area < MIN_BLOB_AREA:
            continue

        # Filter by aspect ratio (reject line fragments)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        # Filter by density (reject sparse noise)
        bbox_area = width * height
        density = area / bbox_area if bbox_area > 0 else 0
        if density < MIN_COMPONENT_DENSITY:
            continue

        valid_count += 1
        total_valid_area += area

    # Calculate ink ratio
    ink_ratio = total_valid_area / (h * w)

    return valid_count, ink_ratio

def determine_has_content(component_count: int, ink_ratio: float) -> bool:
    """Decision logic: combined thresholds."""
    # Primary condition: sufficient valid components
    if component_count >= COMPONENT_COUNT_THRESHOLD:
        return True

    # Fallback condition: high ink ratio (dense content, few large blobs)
    if ink_ratio > INK_RATIO_THRESHOLD:
        return True

    return False
```

---

## 5. OpenCV Performance Optimization

### Decision
Use CPU-based OpenCV (not GPU), pre-allocate work buffers, toggle debug image saving via environment variable, and employ PNG compression level 6 for balanced I/O performance.

### Rationale
- **CPU is Optimal for This Use Case**:
  - ROIs are small (typically 100-300 pixels per side)
  - GPU data transfer overhead exceeds computation savings for small images
  - Morphological operations and connected components have no GPU acceleration in OpenCV
  - Existing codebase is CPU-based (avoid adding CUDA dependency)
- **Pre-Allocated Buffers**: Reduce memory allocation overhead in batch processing
- **Selective Debug Saving**: Saving 6 PNG files per ROI per document adds ~20-30ms; toggle off in production
- **PNG Compression Level 6**: Balances write speed (level 0=fastest) and file size (level 9=smallest)

### Performance Targets (from spec)
- Preprocessing per ROI: <100ms (6 steps × ~15ms average)
- Debug image saving: ~20-30ms per ROI (6 PNG writes)
- Total overhead: ~10-20% increase to existing pipeline time

### Alternatives Considered

| Alternative | Pros | Cons | Rejection Reason |
|------------|------|------|------------------|
| GPU (cv2.cuda) | Faster for large images | High overhead for small ROIs, requires CUDA | Small ROI size makes GPU transfer overhead dominant |
| Parallel multiprocessing | Utilize all CPU cores | Complex synchronization, memory overhead | Not needed (single ROI processes fast enough <100ms) |
| C++ extension | Maximum speed | Development complexity, maintenance burden | Python OpenCV meets performance requirements |
| JPEG debug images | Faster write | Lossy compression, not suitable for debugging | PNG lossless required for threshold tuning |

### Implementation Parameters
```python
# config.py additions
DEBUG_SAVE_INTERMEDIATE_IMAGES = False  # Toggle via environment variable
PNG_COMPRESSION_LEVEL = 6  # Balance speed/size (0=fast/large, 9=slow/small)
REUSE_PREPROCESSOR_INSTANCE = True  # Single instance for batch processing
ENABLE_PARALLEL_PREPROCESSING = False  # Not needed for small ROIs
PRE_ALLOCATE_WORK_BUFFERS = True  # Reuse NumPy arrays
MAX_ROI_DIMENSION = 1000  # Maximum expected ROI size for buffer allocation
```

### Implementation Strategy
```python
class ROIPreprocessor:
    """Optimized preprocessing with buffer reuse."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

        # Pre-allocate work buffers (reused across ROIs)
        if config.PRE_ALLOCATE_WORK_BUFFERS:
            max_dim = config.MAX_ROI_DIMENSION
            self.gray_buffer = np.zeros((max_dim, max_dim), dtype=np.uint8)
            self.binary_buffer = np.zeros((max_dim, max_dim), dtype=np.uint8)

        # Pre-compute morphological kernels (cached by size)
        self.horizontal_kernels = {}
        self.vertical_kernels = {}
        self.noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def get_horizontal_kernel(self, width: int) -> np.ndarray:
        """Get or create horizontal kernel (cached)."""
        if width not in self.horizontal_kernels:
            kw = int(width * self.config.MORPHOLOGY_LINE_RATIO)
            self.horizontal_kernels[width] = cv2.getStructuringElement(
                cv2.MORPH_RECT, (kw, 3)
            )
        return self.horizontal_kernels[width]

    def process_roi(self, doc_roi: np.ndarray, template_roi: np.ndarray) -> PreprocessingResult:
        """Process ROI with optimized memory usage."""
        h, w = doc_roi.shape[:2]

        # Use pre-allocated buffers or create new (fallback)
        if self.config.PRE_ALLOCATE_WORK_BUFFERS:
            gray_doc = self.gray_buffer[:h, :w]
            gray_template = self.gray_buffer[:h, :w]  # Reuse after first use
        else:
            gray_doc = np.zeros((h, w), dtype=np.uint8)
            gray_template = np.zeros((h, w), dtype=np.uint8)

        # Convert to grayscale (in-place)
        cv2.cvtColor(doc_roi, cv2.COLOR_BGR2GRAY, dst=gray_doc)
        cv2.cvtColor(template_roi, cv2.COLOR_BGR2GRAY, dst=gray_template)

        # ... rest of pipeline (reusing buffers)
```

### Environment Variable Integration
```python
# main.py or pipeline.py
import os

DEBUG_MODE = os.getenv('DEBUG_ROI_PREPROCESSING', 'false').lower() == 'true'

# Pass to preprocessor
preprocessor = ROIPreprocessor(
    config=preprocessing_config,
    save_debug_images=DEBUG_MODE,
    output_dir='output/processed_rois' if DEBUG_MODE else None
)
```

---

## Configuration File Complete Reference

Based on all research findings, here is the complete configuration to be added to `vlm_pdf_recognizer/recognition/config.py`:

```python
# ==============================================================================
# ROI PREPROCESSING PIPELINE PARAMETERS (Feature 004-vlm-roi-preprocessing)
# ==============================================================================

# --- Template Difference Configuration ---
# Threshold for binarizing template difference image (grayscale 0-255 scale)
# Lower values detect fainter content but increase noise sensitivity
# Higher values miss faint signatures but reduce false positives
# Recommended range: 15-35, Default: 25
DIFFERENCE_THRESHOLD = 25

# Enable Contrast Limited Adaptive Histogram Equalization before differencing
# Normalizes illumination variations between template and document scans
# Recommended: True for documents scanned under varying lighting conditions
USE_CLAHE_NORMALIZATION = True
CLAHE_CLIP_LIMIT = 2.0  # Contrast limiting (1.0-4.0)
CLAHE_TILE_GRID_SIZE = (8, 8)  # Local equalization grid size

# Median filter kernel size for noise reduction before template difference
# 3 = light filtering, 5 = stronger filtering (may blur faint content)
MEDIAN_FILTER_SIZE = 3

# --- HSV Saturation Preprocessing ---
# Minimum saturation threshold to detect colored content (stamps, seals)
# Range: 0-255, Default: 15 (approximately 6% saturation)
# Lower values detect very faint stamps but may trigger on paper color variation
SATURATION_MIN_THRESHOLD = 15

# Saturation boost factor to amplify faint colored stamps
# Multiplies saturation channel values (e.g., 2.0 doubles saturation)
# Range: 1.0-3.0, Default: 2.0
# Higher values make faint stamps more visible but may over-amplify noise
SATURATION_BOOST_FACTOR = 2.0

# Blending ratio for saturation+value channel combination
# 0.7 = 70% saturation (color), 30% inverted value (dark content)
# Captures both colored stamps and black ink
SATURATION_VALUE_BLEND_RATIO = 0.7

# --- Morphological Line Removal ---
# Kernel size as ratio of ROI dimension for line removal
# E.g., 0.3 on 300px wide ROI → 90px wide kernel
# Range: 0.2-0.5, Default: 0.3
# Lower values: safer for signatures, leaves more line fragments
# Higher values: removes more lines, risks removing signature strokes
MORPHOLOGY_LINE_RATIO = 0.3

# Minimum blob area (pixels) to protect from line removal
# Large blobs (signatures, stamps) are preserved during morphological operations
# Default: 500 pixels (approximately 20x25 pixel region)
MIN_SIGNATURE_AREA = 500

# Enable multi-scale morphological operations (experimental)
# Applies multiple kernel sizes and combines results
# Slower but more robust to varying line widths
USE_MULTI_SCALE_MORPHOLOGY = False
MULTI_SCALE_RATIOS = [0.25, 0.3, 0.35]  # Multiple kernel size ratios

# --- Connected Components Filtering ---
# Minimum area (pixels) for a valid connected component
# Filters out noise and salt-and-pepper artifacts
# Range: 5-100, Default: 20
# Adaptive formula: max(20, ROI_area * 0.001)
MIN_BLOB_AREA = 20

# Minimum number of valid connected components to consider content present
# Range: 1-10, Default: 3
# Signature strokes: 3-15 components
# Stamp characters: 5-50 components
# Form line residue: 10-100+ tiny fragments
COMPONENT_COUNT_THRESHOLD = 3

# Maximum aspect ratio (width/height or height/width) for valid component
# Filters out elongated line fragments
# Range: 5-15, Default: 10
# Signatures: aspect ratio 1-5
# Line fragments: aspect ratio >10
MAX_ASPECT_RATIO = 10

# Minimum component density (component_area / bounding_box_area)
# Filters out sparse noise clusters
# Range: 0.2-0.5, Default: 0.3
# Dense signatures: density 30-80%
# Sparse noise: density <30%
MIN_COMPONENT_DENSITY = 0.3

# --- Feature Extraction Thresholds ---
# Minimum ink ratio (non-zero pixels / total pixels) to consider content present
# Range: 0.001-0.02, Default: 0.005 (0.5%)
# Used as fallback if component count is low but total area is high
INK_RATIO_THRESHOLD = 0.005

# Use 8-connectivity for connected components (connects diagonal pixels)
# True: Better for detecting coherent signature strokes
# False: 4-connectivity (only horizontal/vertical neighbors)
USE_8_CONNECTIVITY = True

# --- Performance Configuration ---
# Save intermediate preprocessing images for debugging
# WARNING: Adds ~20-30ms per ROI, generates ~8MB per document
# Set via environment variable: DEBUG_ROI_PREPROCESSING=true
DEBUG_SAVE_INTERMEDIATE_IMAGES = False

# PNG compression level for debug images
# Range: 0-9 (0=fastest/largest, 9=slowest/smallest)
# Default: 6 (balanced)
PNG_COMPRESSION_LEVEL = 6

# Reuse ROIPreprocessor instance for batch processing (recommended)
# Pre-allocates buffers and caches morphological kernels
REUSE_PREPROCESSOR_INSTANCE = True

# Enable parallel preprocessing across multiple ROIs (experimental)
# Not recommended for small ROIs (<500px) due to overhead
ENABLE_PARALLEL_PREPROCESSING = False

# Pre-allocate work buffers for memory efficiency
# Reduces allocation overhead in batch processing
PRE_ALLOCATE_WORK_BUFFERS = True

# Maximum expected ROI dimension for buffer allocation
# Buffers sized to (MAX_ROI_DIMENSION × MAX_ROI_DIMENSION)
MAX_ROI_DIMENSION = 1000
```

---

## Next Steps

1. **Implement ROIPreprocessor class** using researched techniques
2. **Integrate into vlm_recognizer.py** preprocessing pipeline
3. **Test with real documents** and tune thresholds based on actual results
4. **Benchmark performance** to validate <100ms per ROI target
5. **Generate debug images** for threshold tuning and edge case debugging

All technical decisions are now documented with rationale, alternatives considered, and concrete implementation parameters.
