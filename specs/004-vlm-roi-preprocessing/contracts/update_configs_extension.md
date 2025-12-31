# update_configs.py Extension Contract

**Feature**: 004-vlm-roi-preprocessing
**Module**: `update_configs.py`
**Created**: 2025-12-31

## Overview

This document defines the modifications to `update_configs.py` to generate blank template ROI images during template configuration.

---

## New Function: extract_blank_template_rois

```python
def extract_blank_template_rois(
    template_id: str,
    template_image: np.ndarray,
    rois: list[dict]
) -> dict[str, np.ndarray]:
    """
    Extract blank template ROI images from template source image.

    Args:
        template_id: Template identifier (e.g., 'contractor_1')
        template_image: Blank template image, BGR format, uint8, shape (H, W, 3)
        rois: List of ROI dictionaries from config.json, each with:
              - 'id': Field identifier (str)
              - 'coordinates': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (list of 4 points)

    Returns:
        Dictionary mapping field_id -> ROI image (BGR uint8 np.ndarray)

    Raises:
        ValueError: If template_image is invalid (wrong shape, dtype)
        ValueError: If ROI coordinates are invalid (out of bounds, not 4 points)
    """
```

**Contract**:
- MUST extract rectangular ROI for each field using coordinates
- MUST perform perspective transform if ROI is not axis-aligned
- MUST return BGR uint8 images matching input template format
- MUST validate all coordinates are within template image bounds
- MUST handle rotated/skewed ROIs via perspective warp

**Implementation Pattern** (reuse existing ROI extraction logic):
```python
# Similar to existing roi_extractor.extract_rois logic
for roi_config in rois:
    field_id = roi_config['id']
    coords = np.array(roi_config['coordinates'], dtype=np.float32)

    # Extract ROI (perspective transform if needed)
    roi_image = extract_roi_from_coordinates(template_image, coords)

    blank_rois[field_id] = roi_image
```

---

## New Function: save_blank_template_rois

```python
def save_blank_template_rois(
    template_id: str,
    blank_rois: dict[str, np.ndarray],
    output_dir: Path | str
) -> None:
    """
    Save blank template ROI images to disk.

    Args:
        template_id: Template identifier
        blank_rois: Dictionary mapping field_id -> ROI image
        output_dir: Base output directory (e.g., 'data/')

    Side Effects:
        - Creates directory: {output_dir}/{template_id}/blank_rois/
        - Writes PNG files: {field_id}.png for each ROI
    """
```

**Contract**:
- MUST create directory structure: {output_dir}/{template_id}/blank_rois/
- MUST save each ROI as PNG file: {output_dir}/{template_id}/blank_rois/{field_id}.png
- MUST use PNG lossless compression
- MUST overwrite existing files if re-running configuration
- MUST handle I/O errors gracefully (log error, continue with other ROIs)

**File Naming Convention**:
```
data/
├── contractor_1/
│   ├── blank_rois/
│   │   ├── title.png
│   │   ├── VX1.png
│   │   ├── VX2.png
│   │   ├── year.png
│   │   ├── month.png
│   │   ├── date.png
│   │   ├── company1.png
│   │   ├── signature.png
│   │   └── stamp1.png
│   ├── template.png
│   └── config.json
```

---

## Modified Function: main (update_configs.py)

**Existing Behavior** (preserved):
1. Load LabelMe annotation JSONs from templates/location/{template_id}.json
2. Extract ROI coordinates from annotations
3. Generate SIFT features for template matching
4. Save template.png and config.json

**New Behavior** (added):
5. Extract blank template ROI images using ROI coordinates
6. Save blank template ROI images to data/{template_id}/blank_rois/

**Modified Code Structure**:
```python
def main():
    """Update template configurations and extract blank ROIs."""
    templates_dir = Path("data")
    annotation_dir = Path("templates/location")

    # For each template
    for annotation_file in annotation_dir.glob("*.json"):
        template_id = annotation_file.stem

        # Existing: Load annotation and template image
        with open(annotation_file) as f:
            annotation = json.load(f)

        template_image_path = Path("templates") / annotation['imagePath']
        template_image = cv2.imread(str(template_image_path))

        # Existing: Extract ROI coordinates and SIFT features
        rois = extract_rois_from_annotation(annotation)
        template_features = extract_template_features(template_image)

        # Existing: Save template.png and config.json
        output_dir = templates_dir / template_id
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "template.png"), template_image)
        save_config_json(output_dir / "config.json", rois, template_features)

        # NEW: Extract blank template ROIs
        blank_rois = extract_blank_template_rois(template_id, template_image, rois)

        # NEW: Save blank template ROIs
        save_blank_template_rois(template_id, blank_rois, templates_dir)

        print(f"✓ Template '{template_id}': Generated {len(blank_rois)} blank ROIs")
```

---

## Console Output

**Before** (existing output):
```
Initializing processor...

✓ Template 'contractor_1': Updated successfully
✓ Template 'contractor_2': Updated successfully
✓ Template 'enterprise_1': Updated successfully

Done! Updated 3 template(s).
```

**After** (with blank ROI generation):
```
Initializing processor...

✓ Template 'contractor_1': Updated successfully
  → Generated 13 blank ROI images in data/contractor_1/blank_rois/
✓ Template 'contractor_2': Updated successfully
  → Generated 13 blank ROI images in data/contractor_2/blank_rois/
✓ Template 'enterprise_1': Updated successfully
  → Generated 14 blank ROI images in data/enterprise_1/blank_rois/

Done! Updated 3 template(s).
Total blank ROIs generated: 40
```

---

## Error Handling

### Recoverable Errors (Log warning, continue)
- ROI extraction fails for one field → Skip that ROI, continue with others
- Blank ROI save fails for one field → Log error, continue with others
- Template image not found → Skip that template, continue with others

### Non-Recoverable Errors (Exit with error message)
- Annotation JSON parsing fails (invalid format)
- Template image file is corrupted (cannot read)
- Output directory not writable (permission error)

---

## Backward Compatibility

**Guaranteed**: Existing functionality (SIFT feature extraction, config.json generation) is UNCHANGED.

**New files**: blank_rois/ directory and PNG files are NEW additions. Old code that doesn't use preprocessing will simply ignore these files.

**Safe to re-run**: Running update_configs.py overwrites existing blank ROIs. Safe to re-run after template changes.

---

## Testing Contract

### Unit Test: test_extract_blank_template_rois
```python
def test_extract_blank_template_rois():
    """Test blank ROI extraction from template image."""
    # Create test template image (300x400 white image)
    template_image = np.ones((400, 300, 3), dtype=np.uint8) * 255

    # Define test ROI coordinates (simple rectangle)
    rois = [
        {'id': 'field1', 'coordinates': [[10, 10], [110, 10], [110, 60], [10, 60]]}
    ]

    # Extract blank ROIs
    blank_rois = extract_blank_template_rois('test_template', template_image, rois)

    # Verify extraction
    assert 'field1' in blank_rois
    assert blank_rois['field1'].shape == (50, 100, 3)  # Height 50, Width 100
    assert blank_rois['field1'].dtype == np.uint8
    assert np.all(blank_rois['field1'] == 255)  # All white (blank)
```

### Integration Test: test_update_configs_generates_blank_rois
```python
def test_update_configs_generates_blank_rois(tmp_path):
    """Test that update_configs.py generates blank ROI files."""
    # Run update_configs.py
    subprocess.run(['python', 'update_configs.py'], check=True)

    # Verify blank_rois directories created
    for template_id in ['contractor_1', 'contractor_2', 'enterprise_1']:
        blank_rois_dir = Path('data') / template_id / 'blank_rois'
        assert blank_rois_dir.exists()
        assert blank_rois_dir.is_dir()

        # Verify PNG files exist
        png_files = list(blank_rois_dir.glob('*.png'))
        assert len(png_files) > 0  # At least one blank ROI generated

        # Verify images are valid
        for png_file in png_files:
            img = cv2.imread(str(png_file))
            assert img is not None
            assert img.dtype == np.uint8
            assert len(img.shape) == 3  # BGR format
```

---

## Performance Contract

**Time**: Blank ROI extraction should add <2 seconds to total update_configs.py execution time for all 3 templates.

**Breakdown**:
- ROI extraction: ~10ms per ROI (reuses existing extraction logic)
- PNG saving: ~5ms per ROI (lossless compression)
- Total for 40 ROIs: ~600ms

**Space**: Blank ROI images total ~2MB for all templates (40 ROIs × ~50KB each).

---

## Dependencies

**Existing Dependencies** (already in update_configs.py):
- OpenCV (cv2)
- NumPy
- pathlib
- json

**No New Dependencies Required**
