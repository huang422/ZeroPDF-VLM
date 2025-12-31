# ROI Preprocessor Interface Contract

**Feature**: 004-vlm-roi-preprocessing
**Module**: `vlm_pdf_recognizer.recognition.roi_preprocessor`
**Created**: 2025-12-31

## Overview

This document defines the interface contract for the `ROIPreprocessor` class, which executes the 6-step image preprocessing pipeline for ROI content detection.

---

## Class: ROIPreprocessor

### Constructor

```python
class ROIPreprocessor:
    def __init__(
        self,
        config: PreprocessingConfig,
        save_debug_images: bool = False,
        output_dir: str | Path | None = None
    ) -> None:
        """
        Initialize ROI preprocessing pipeline.

        Args:
            config: Preprocessing configuration parameters from config.py
            save_debug_images: Whether to save intermediate images for debugging
            output_dir: Directory path for saving debug images (required if save_debug_images=True)

        Raises:
            ValueError: If save_debug_images=True but output_dir is None
            ValueError: If config parameters are outside valid ranges
        """
```

**Contract**:
- MUST validate all config parameters are within specified ranges
- MUST pre-allocate work buffers if `config.PRE_ALLOCATE_WORK_BUFFERS` is True
- MUST create output_dir if it doesn't exist when save_debug_images=True
- MUST cache morphological kernels for reuse across ROI processing calls

**Postconditions**:
- Preprocessor ready to process ROIs without further initialization
- Kernel cache initialized (empty dict)
- Work buffers allocated if configured

---

### Method: preprocess_roi

```python
def preprocess_roi(
    self,
    doc_roi_image: np.ndarray,
    blank_template_roi: np.ndarray,
    field_id: str,
    document_name: str | None = None
) -> PreprocessingResult:
    """
    Execute 6-step preprocessing pipeline on document ROI.

    Args:
        doc_roi_image: Document ROI image, BGR format, uint8, shape (H, W, 3)
        blank_template_roi: Blank template ROI image, BGR format, uint8, shape (H, W, 3)
        field_id: Field identifier for logging and debug image naming
        document_name: Document name for debug image directory structure (optional)

    Returns:
        PreprocessingResult with has_content determination and metrics

    Raises:
        ValueError: If doc_roi_image or blank_template_roi are invalid (wrong shape, dtype)
        RuntimeError: If preprocessing pipeline fails catastrophically (image corruption)

    Pipeline Steps:
        1. HSV Saturation Extraction: Convert to HSV, extract/boost saturation channel
        2. Template Difference: Compute absolute difference, apply binary threshold
        3. Horizontal Line Removal: Morphological opening with horizontal kernel
        4. Vertical Line Removal: Morphological opening with vertical kernel
        5. Noise Removal: Small blob removal via connected components
        6. Feature Extraction & Decision: Calculate ink_ratio, component_count, determine has_content
    """
```

**Contract**:
- MUST resize doc_roi_image to match blank_template_roi dimensions if sizes differ
- MUST log warning if dimension mismatch >10%
- MUST execute all 6 steps in order (no skipping steps)
- MUST save intermediate images (01_saturation.png through 06_final_binary.png) if save_debug_images=True
- MUST save metadata.json with metrics and decision reasoning if save_debug_images=True
- MUST return valid PreprocessingResult even if errors occur (set has_content=None, populate error_message)
- MUST NOT raise exceptions for recoverable errors (noise, poor image quality)
- MAY raise ValueError/RuntimeError only for invalid input (wrong dtype, corrupted image data)

**Preconditions**:
- doc_roi_image and blank_template_roi must be valid BGR images (H, W, 3) with dtype uint8
- field_id must be non-empty string
- If save_debug_images=True, output_dir must have been set in constructor

**Postconditions**:
- PreprocessingResult.has_content is bool (True/False) if pipeline succeeded, None if failed
- If has_content is not None, preprocessing metrics (ink_ratio, component_count) are populated
- If save_debug_images=True, 6 PNG files and 1 metadata.json are written to disk
- Processing time is recorded in PreprocessingResult.processing_time_ms

**Performance Guarantees**:
- MUST complete in <100ms per ROI on standard CPU (excluding debug image I/O)
- Debug image saving MAY add up to 30ms additional time
- Memory usage MUST NOT exceed 50MB per ROI (including intermediate images)

---

### Method: _extract_saturation_channel (Internal)

```python
def _extract_saturation_channel(
    self,
    roi_image: np.ndarray
) -> np.ndarray:
    """
    Step 1: Extract and enhance saturation channel for faint stamp preservation.

    Args:
        roi_image: Input ROI image, BGR format, uint8, shape (H, W, 3)

    Returns:
        Single-channel image (H, W) with enhanced saturation, uint8
    """
```

**Contract**:
- MUST convert BGR to HSV color space
- MUST extract saturation channel (S from HSV)
- MUST apply SATURATION_BOOST_FACTOR amplification
- MUST clip values to 0-255 range
- MUST optionally blend with inverted value channel (V) using SATURATION_VALUE_BLEND_RATIO
- MUST return uint8 single-channel image

---

### Method: _compute_template_difference (Internal)

```python
def _compute_template_difference(
    self,
    doc_saturation: np.ndarray,
    template_saturation: np.ndarray
) -> np.ndarray:
    """
    Step 2: Compute absolute difference and apply binary threshold.

    Args:
        doc_saturation: Document saturation image, grayscale uint8, shape (H, W)
        template_saturation: Template saturation image, grayscale uint8, shape (H, W)

    Returns:
        Binary difference image (H, W), uint8, values 0 or 255
    """
```

**Contract**:
- MUST optionally apply CLAHE normalization if USE_CLAHE_NORMALIZATION=True
- MUST optionally apply median filtering if MEDIAN_FILTER_SIZE > 0
- MUST compute absolute difference: |doc - template|
- MUST apply binary threshold at DIFFERENCE_THRESHOLD
- MUST return binary image (0=no difference, 255=difference)

---

### Method: _remove_form_lines (Internal)

```python
def _remove_form_lines(
    self,
    binary_image: np.ndarray
) -> np.ndarray:
    """
    Step 3-4: Remove horizontal and vertical form lines via morphological opening.

    Args:
        binary_image: Binary difference image, uint8, shape (H, W), values 0 or 255

    Returns:
        Binary image with lines removed, uint8, shape (H, W)
    """
```

**Contract**:
- MUST create horizontal kernel: width = ROI_width * MORPHOLOGY_LINE_RATIO, height = 3
- MUST create vertical kernel: width = 3, height = ROI_height * MORPHOLOGY_LINE_RATIO
- MUST apply morphological opening (erosion + dilation) for horizontal lines
- MUST apply morphological opening (erosion + dilation) for vertical lines
- MUST optionally apply multi-scale if USE_MULTI_SCALE_MORPHOLOGY=True
- MUST cache kernels by size to avoid recomputation
- MUST return binary image with lines removed

---

### Method: _remove_noise (Internal)

```python
def _remove_noise(
    self,
    binary_image: np.ndarray
) -> np.ndarray:
    """
    Step 4 (continued): Remove salt-and-pepper noise and small blobs.

    Args:
        binary_image: Binary image with lines removed, uint8, shape (H, W)

    Returns:
        Binary image with noise removed, uint8, shape (H, W)
    """
```

**Contract**:
- MUST apply morphological opening with small kernel (3x3) for salt-and-pepper noise
- MUST extract connected components with 8-connectivity if USE_8_CONNECTIVITY=True
- MUST remove components with area < MIN_BLOB_AREA
- MUST return cleaned binary image

---

### Method: _extract_features (Internal)

```python
def _extract_features(
    self,
    binary_image: np.ndarray
) -> tuple[int, float]:
    """
    Step 5: Extract ink ratio and valid component count.

    Args:
        binary_image: Cleaned binary image, uint8, shape (H, W)

    Returns:
        Tuple of (component_count, ink_ratio)
        - component_count: Number of valid connected components (int)
        - ink_ratio: Ratio of non-zero pixels to total pixels (float, 0.0-1.0)
    """
```

**Contract**:
- MUST extract connected components with 8-connectivity
- MUST filter components by:
  - Minimum area (MIN_BLOB_AREA)
  - Maximum aspect ratio (MAX_ASPECT_RATIO)
  - Minimum density (MIN_COMPONENT_DENSITY)
- MUST count valid components
- MUST calculate ink ratio: total_valid_area / total_pixels
- MUST return (component_count, ink_ratio) tuple

---

### Method: _determine_has_content (Internal)

```python
def _determine_has_content(
    self,
    component_count: int,
    ink_ratio: float
) -> tuple[bool, str]:
    """
    Step 6: Decision logic to determine content presence.

    Args:
        component_count: Number of valid connected components
        ink_ratio: Ink ratio (0.0-1.0)

    Returns:
        Tuple of (has_content, reasoning)
        - has_content: True if content detected, False otherwise
        - reasoning: Human-readable explanation of decision
    """
```

**Contract**:
- MUST return True if component_count >= COMPONENT_COUNT_THRESHOLD
- MUST return True if ink_ratio > INK_RATIO_THRESHOLD (fallback for dense content)
- MUST return False if both conditions fail
- MUST provide reasoning string in format: "ink_ratio=X.XXX ? threshold=Y.YYY AND component_count=N ? threshold=M, has_content=True/False"

---

### Method: _save_intermediate_images (Internal)

```python
def _save_intermediate_images(
    self,
    intermediate_images: dict[str, np.ndarray],
    field_id: str,
    document_name: str
) -> None:
    """
    Save intermediate preprocessing images for debugging.

    Args:
        intermediate_images: Dict mapping step_name -> image
        field_id: Field identifier
        document_name: Document name

    Side Effects:
        - Creates directory: output_dir/{document_name}/{field_id}/
        - Writes 6 PNG files: 01_saturation.png through 06_final_binary.png
    """
```

**Contract**:
- MUST create directory structure: {output_dir}/{document_name}/{field_id}/
- MUST save images with sequential naming: 01_step_name.png, 02_step_name.png, etc.
- MUST use PNG compression level from config.PNG_COMPRESSION_LEVEL
- MUST handle I/O errors gracefully (log error, do not raise exception)

---

### Method: _save_metadata (Internal)

```python
def _save_metadata(
    self,
    field_id: str,
    document_name: str,
    result: PreprocessingResult
) -> None:
    """
    Save preprocessing metadata JSON for debugging.

    Args:
        field_id: Field identifier
        document_name: Document name
        result: PreprocessingResult with metrics

    Side Effects:
        - Writes {output_dir}/{document_name}/{field_id}/metadata.json
    """
```

**Contract**:
- MUST save JSON file with structure:
  ```json
  {
    "field_id": "...",
    "has_content": true/false/null,
    "ink_ratio": 0.012,
    "component_count": 5,
    "processing_time_ms": 45.2,
    "thresholds_used": { ... },
    "decision_reasoning": "...",
    "error_message": null
  }
  ```
- MUST include all threshold values from config
- MUST include decision reasoning string
- MUST handle I/O errors gracefully

---

## Data Types

### PreprocessingConfig (from config.py)

```python
@dataclass
class PreprocessingConfig:
    """Configuration parameters for preprocessing pipeline."""
    DIFFERENCE_THRESHOLD: int = 25
    USE_CLAHE_NORMALIZATION: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)
    MEDIAN_FILTER_SIZE: int = 3
    SATURATION_MIN_THRESHOLD: int = 15
    SATURATION_BOOST_FACTOR: float = 2.0
    SATURATION_VALUE_BLEND_RATIO: float = 0.7
    MORPHOLOGY_LINE_RATIO: float = 0.3
    MIN_SIGNATURE_AREA: int = 500
    USE_MULTI_SCALE_MORPHOLOGY: bool = False
    MULTI_SCALE_RATIOS: list[float] = field(default_factory=lambda: [0.25, 0.3, 0.35])
    MIN_BLOB_AREA: int = 20
    COMPONENT_COUNT_THRESHOLD: int = 3
    MAX_ASPECT_RATIO: int = 10
    MIN_COMPONENT_DENSITY: float = 0.3
    INK_RATIO_THRESHOLD: float = 0.005
    USE_8_CONNECTIVITY: bool = True
    DEBUG_SAVE_INTERMEDIATE_IMAGES: bool = False
    PNG_COMPRESSION_LEVEL: int = 6
    REUSE_PREPROCESSOR_INSTANCE: bool = True
    ENABLE_PARALLEL_PREPROCESSING: bool = False
    PRE_ALLOCATE_WORK_BUFFERS: bool = True
    MAX_ROI_DIMENSION: int = 1000
```

### PreprocessingResult (Output)

```python
@dataclass
class PreprocessingResult:
    """Output of preprocessing pipeline for a single ROI."""
    field_id: str
    has_content: bool | None  # True=content, False=empty, None=error
    ink_ratio: float | None  # 0.0-1.0, None if error
    component_count: int | None  # >=0, None if error
    processing_time_ms: float  # Always populated
    error_message: str | None  # None if successful
    intermediate_images: dict[str, np.ndarray] | None  # Only if save_debug_images=True
```

---

## Error Handling Contract

### Recoverable Errors (Return has_content=None, log warning)
- Dimension mismatch between doc_roi and template_roi (resize and continue)
- Poor image quality (low contrast, noise) → Best effort processing
- Missing morphological kernel in cache → Compute and cache
- Debug image save failure → Log error, continue processing

### Non-Recoverable Errors (Raise exception)
- Invalid image format (not BGR uint8)
- Corrupted image data (cannot read pixels)
- Invalid config parameters (out of range)
- output_dir not set when save_debug_images=True

---

## Performance Contract

### Timing Guarantees
- Step 1 (Saturation): <10ms
- Step 2 (Difference): <15ms (with CLAHE)
- Step 3 (H-line removal): <20ms
- Step 4 (V-line removal): <20ms
- Step 5 (Noise removal): <10ms
- Step 6 (Features): <15ms
- **Total (without debug I/O): <100ms**
- Debug image saving: +20-30ms

### Memory Contract
- Work buffers: <10MB per preprocessor instance
- Intermediate images: ~5MB (6 images × ~500KB each)
- Kernel cache: <1MB
- **Peak usage per ROI: <50MB**

---

## Thread Safety

**NOT THREAD-SAFE**: ROIPreprocessor reuses internal work buffers. Do not call `preprocess_roi` from multiple threads simultaneously on the same instance.

**Recommended Usage**:
- Single-threaded batch processing: Reuse one instance
- Multi-threaded batch processing: Create one instance per thread

---

## Backward Compatibility

This interface is NEW in feature 004. No backward compatibility requirements.

Future API changes MUST maintain signature compatibility or provide deprecated wrappers.
