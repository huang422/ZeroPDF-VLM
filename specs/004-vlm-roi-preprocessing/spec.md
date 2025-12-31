# Feature Specification: VLM-Assisted ROI Content Detection with Image Preprocessing

**Feature Branch**: `004-vlm-roi-preprocessing`
**Created**: 2025-12-31
**Status**: Draft
**Input**: User description: "VLM輔助辨識功能roi影像處理"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Template ROI Baseline Generation (Priority: P1)

During system initialization, administrators run the template configuration script to generate blank template ROI images for each field position. These blank template ROIs serve as reference baselines to differentiate pre-printed content (form borders, fixed text) from user-added content (signatures, stamps, handwriting).

**Why this priority**: Foundation for the entire detection system. Without blank template ROI references, the preprocessing pipeline cannot distinguish between pre-printed form elements and actual user content.

**Independent Test**: Run update_configs.py with template images, verify that blank template ROI images are saved to data/{template_id}/blank_rois/{field_id}.png for each field, and can be loaded for comparison.

**Acceptance Scenarios**:

1. **Given** contractor_1 template with 13 defined ROI coordinates, **When** configuration script executes, **Then** 13 blank template ROI images are extracted and saved with field IDs matching the configuration file.

2. **Given** an updated template with modified ROI coordinates, **When** configuration re-runs, **Then** existing blank template ROIs are replaced with newly extracted ROIs matching the updated positions.

3. **Given** a template with various ROI types (text fields, signature boxes, stamp areas), **When** blank ROIs are generated, **Then** each ROI preserves pre-printed elements like borders, baseline text, and fixed labels.

---

### User Story 2 - ROI Content Detection via Template Difference Preprocessing (Priority: P1)

When processing a document, the system preprocesses each extracted ROI by comparing it against the corresponding blank template ROI using a 6-step image processing pipeline. This pipeline removes pre-printed form elements, filters noise, and produces a binary has_content decision (True/False) indicating whether user content exists in that field.

**Why this priority**: Core differentiator from existing VLM-only approach. The preprocessing pipeline provides pixel-level content detection that is more reliable for visual elements (stamps, signatures) than VLM zero-shot inference alone.

**Independent Test**: Process a test document with mixed filled/unfilled fields, verify that the preprocessing pipeline correctly identifies filled fields (has_content=True) and empty fields (has_content=False), with intermediate processing images saved to output/processed_rois/ for inspection.

**Acceptance Scenarios**:

1. **Given** a document ROI containing only the pre-printed form border (no signature added), **When** template difference preprocessing runs, **Then** the difference image shows minimal content, has_content is False, and no user content is detected.

2. **Given** a document ROI with a faint red stamp (low saturation), **When** HSV preprocessing and template difference are applied, **Then** the stamp is preserved through saturation channel extraction, has_content is True, and the stamp is detected.

3. **Given** a document ROI with handwritten signature, **When** preprocessing pipeline executes all 6 steps, **Then** signature strokes survive noise removal and morphological filtering, final connected components count exceeds threshold, and has_content is True.

4. **Given** a document ROI with residual form line fragments after template difference, **When** morphological opening for horizontal/vertical lines runs, **Then** long straight lines are removed, signature/stamp content remains, and has_content determination is accurate.

---

### User Story 3 - Integrated VLM Recognition with Preprocessed Content Flags (Priority: P1)

After preprocessing determines has_content for each ROI, the system passes all ROIs (regardless of preprocessing result) to the VLM for content extraction. The VLM extracts text/numbers from fields, while the preprocessing result provides a more reliable has_content flag for validation logic.

**Why this priority**: Maintains full VLM functionality for content extraction while improving content presence detection. Users benefit from both pixel-based detection accuracy and VLM semantic understanding.

**Independent Test**: Process a document where preprocessing detects content in 10 fields, verify that VLM runs on all fields, extracts text where present, and output contains both VLM_has_content and AUX_has_content columns with potentially different values for debugging.

**Acceptance Scenarios**:

1. **Given** a field where preprocessing detects content (has_content=True), **When** VLM inference runs, **Then** VLM extracts the text/number content and VLM_has_content is populated based on VLM's own judgment.

2. **Given** a field where preprocessing finds no content (has_content=False), **When** VLM inference still runs, **Then** VLM result is captured for comparison, but validation logic prioritizes the preprocessing result.

3. **Given** a title field that skips preprocessing, **When** VLM recognition executes, **Then** title content is extracted, AUX_has_content is None, and VLM_has_content is not used for validation.

---

### User Story 4 - Auxiliary-Priority Validation Logic (Priority: P1)

The system calculates overall document validation status using the preprocessing has_content results as the primary input, while preserving the existing validation logic rules (VX1 priority, date field OR, other fields AND). This replaces the previous VLM-based or auxiliary comparison-based validation with preprocessing-based validation.

**Why this priority**: Accuracy improvement and consistency. The preprocessing pipeline is deterministic and explainable, providing more reliable validation than VLM hallucination-prone inference or simple SIFT feature matching.

**Independent Test**: Process documents with known completion status, verify that results field correctly reflects validation using preprocessing has_content values, matching existing logic (VX1=True means results=False, all date fields False means results=False, any other field False means results=False).

**Acceptance Scenarios**:

1. **Given** a document where VX1 preprocessing has_content is True (disagreement checkbox marked), **When** validation logic runs, **Then** results is immediately set to False regardless of other field values.

2. **Given** a document where all date fields (year, month, date) have preprocessing has_content=False, **When** validation calculates results, **Then** results is False because date OR condition fails.

3. **Given** a document where at least one date field has preprocessing has_content=True and all other required fields have preprocessing has_content=True, **When** validation runs, **Then** results is True.

4. **Given** a document where one required field (e.g., company1) has preprocessing has_content=False, **When** validation logic executes, **Then** results is False because the AND condition for other fields fails.

---

### User Story 5 - Color-Coded Visualization with Preprocessing Results (Priority: P2)

The system generates visualization images with ROI bounding boxes colored based on preprocessing has_content values (green for True, red for False). This provides visual quick-scan of document completion status based on the preprocessing pipeline's deterministic pixel analysis.

**Why this priority**: User experience enhancement. Visual feedback allows rapid batch validation and debugging, with preprocessing results being more trustworthy for visual inspection than VLM text descriptions.

**Independent Test**: Process a document and verify that visualization image shows green boxes around filled fields (preprocessing detected content) and red boxes around empty fields (preprocessing detected no content), with labels indicating preprocessing source.

**Acceptance Scenarios**:

1. **Given** a document with 10 completed fields and 3 empty fields, **When** visualization is generated, **Then** 10 ROI boxes are drawn in green (preprocessing has_content=True) and 3 boxes in red (preprocessing has_content=False).

2. **Given** a field where preprocessing failed to run (error case), **When** visualization renders, **Then** the ROI box defaults to neutral color (blue) to indicate uncertainty, with error logged.

3. **Given** a title field that has no preprocessing, **When** visualization is drawn, **Then** the title ROI box uses original template color since it's not subject to validation.

---

### User Story 6 - Checkbox Recognition Preservation (Priority: P1)

VX1 and VX2 checkbox fields continue using the existing heuristic pixel check method without preprocessing pipeline application, as the current pixel counting approach already provides reliable checkbox detection and preprocessing would add unnecessary complexity.

**Why this priority**: Regression prevention. Checkbox recognition works well with existing heuristic; preprocessing pipeline is designed for signature/stamp/text fields with more complex content patterns.

**Independent Test**: Process documents with checked/unchecked VX1 and VX2 boxes, verify that recognition results match existing behavior (heuristic pixel counting), preprocessing pipeline is skipped for checkboxes, and validation logic uses heuristic results for VX1/VX2.

**Acceptance Scenarios**:

1. **Given** a document with VX1 checkbox checked, **When** recognition runs, **Then** existing heuristic detects checkmark, preprocessing pipeline is skipped, and VX1.has_content is True from heuristic.

2. **Given** a document with VX2 checkbox unchecked, **When** recognition executes, **Then** heuristic returns False, preprocessing is bypassed, and VX2.has_content is False.

3. **Given** checkbox results, **When** validation logic runs, **Then** VX1 and VX2 use heuristic has_content values rather than preprocessing has_content.

---

### User Story 7 - Processed ROI Image Export (Priority: P2)

The system saves intermediate and final preprocessing images for each ROI to output/processed_rois/{document_name}/{field_id}/ directory, including difference images, filtered images, and binary output images. This enables pipeline debugging, threshold tuning, and quality verification.

**Why this priority**: Debugging and transparency. Users and developers can visually inspect each preprocessing step to understand why content was detected or missed, and tune thresholds for their specific form types.

**Independent Test**: Process a document with preprocessing enabled, verify that output/processed_rois/ contains subdirectories for each document with step-by-step images for each field (01_grayscale.png, 02_difference.png, 03_morphology.png, 04_noise_removed.png, 05_binary.png).

**Acceptance Scenarios**:

1. **Given** a document with 13 ROI fields, **When** preprocessing completes, **Then** output directory contains 13 subdirectories with 5+ intermediate images each showing preprocessing stages.

2. **Given** a field where preprocessing detected content, **When** inspecting saved images, **Then** difference image shows clear content regions, morphology image shows line removal, and binary image shows final detected content blobs.

3. **Given** a field where preprocessing failed to detect faint content, **When** reviewing saved images, **Then** user can identify which step lost the content (threshold too high, noise removal too aggressive, etc.) and adjust config parameters.

---

### Edge Cases

- **Blank template quality variations**: What happens when the blank template has slight printing defects or quality variations? The template difference step may produce false positives; solution is to use multiple blank template samples and combine their difference results with OR logic.

- **Partial content filling**: How does system handle very light signatures or extremely faint stamps? HSV saturation channel preprocessing helps preserve faint colors; threshold tuning (DIFFERENCE_THRESHOLD, INK_RATIO_THRESHOLD) can be adjusted based on test results.

- **Template alignment errors**: What happens when document alignment fails and ROI positions are incorrect? Template difference will fail to cancel out pre-printed content (wrong regions compared); system should detect high difference scores even in "blank" fields and log alignment warning.

- **Rotated or skewed signatures**: How does preprocessing handle rotated signatures within aligned ROIs? Morphological operations use multiple kernel orientations to avoid removing rotated strokes; connected components analysis is rotation-invariant.

- **Overlapping content**: What happens when a stamp overlaps pre-printed text? Template difference preserves the stamp overlay while removing base text; the overlapping region appears as added content (correct behavior).

- **Color stamps on color forms**: How does system differentiate color stamps from color-printed form elements? Saturation channel analysis in HSV space detects stamp saturation differences; template difference in grayscale catches intensity changes.

- **Multi-page PDFs with mixed templates**: How does system select correct blank template ROI for each page? Match based on template_id from alignment phase; each page's ROIs compare against its matched template's blank ROIs.

- **Missing blank template ROIs**: How does system behave if blank ROI images are not found for a template? Log warning, skip preprocessing for that template, fall back to VLM-only recognition to maintain functionality.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extend update_configs.py to extract blank template ROI images from template source images using configured ROI coordinates, saving ROI images to data/{template_id}/blank_rois/{field_id}.png for each field.

- **FR-002**: System MUST implement a 6-step ROI preprocessing pipeline for content detection:
  - Step 1 (Preprocessing): Convert ROI to HSV color space, extract saturation channel to preserve faint color stamps, return single-channel image for subsequent processing
  - Step 2 (Template Difference): Convert both sample and template ROIs to grayscale, compute absolute difference, apply binary threshold to isolate added content, produce binary difference image
  - Step 3 (Structural Filtering): Apply morphological opening with horizontal kernel (width = ROI_width * 0.3, height = 3) to remove horizontal lines, apply morphological opening with vertical kernel (width = 3, height = ROI_height * 0.3) to remove vertical lines, preserve signatures/stamps (irregular shapes not affected by line-oriented kernels)
  - Step 4 (Noise Removal): Apply morphological opening with small kernel (3x3) to remove salt-and-pepper noise, apply connected components analysis to remove blobs smaller than MIN_BLOB_AREA (configurable, default 20 pixels)
  - Step 5 (Feature Extraction): Calculate ink_ratio = non-zero_pixels / total_pixels, count connected components with area > MIN_BLOB_AREA, return (ink_ratio, component_count)
  - Step 6 (Decision Logic): has_content = True if (ink_ratio > INK_RATIO_THRESHOLD AND component_count >= COMPONENT_COUNT_THRESHOLD), else has_content = False

- **FR-003**: System MUST make preprocessing threshold parameters configurable via vlm_pdf_recognizer/recognition/config.py:
  - DIFFERENCE_THRESHOLD: Threshold for template difference binarization (default 25, range 10-50)
  - INK_RATIO_THRESHOLD: Minimum ink ratio to consider content present (default 0.005 = 0.5%, range 0.001-0.02)
  - COMPONENT_COUNT_THRESHOLD: Minimum connected components for content (default 3, range 1-10)
  - MIN_BLOB_AREA: Minimum pixel area for valid component (default 20, range 5-100)
  - MORPHOLOGY_LINE_RATIO: Ratio of ROI dimension for line removal kernels (default 0.3, range 0.2-0.5)

- **FR-004**: System MUST apply preprocessing pipeline to all non-title, non-checkbox fields (text, number, stamp fields) during VLM recognition workflow, before VLM inference is invoked.

- **FR-005**: System MUST skip preprocessing pipeline for title fields (no validation needed) and checkbox fields (existing heuristic is sufficient), setting preprocessing has_content to None for these field types.

- **FR-006**: System MUST always invoke VLM inference for all non-title fields regardless of preprocessing result, to extract content_text even when preprocessing determines has_content=False (for debugging and comparison purposes).

- **FR-007**: System MUST store both preprocessing has_content and VLM has_content values in RecognitionResult dataclass:
  - Add preprocessing_has_content field (True/False/None)
  - Add preprocessing_metrics field containing (ink_ratio, component_count, processing_time_ms)
  - Preserve existing VLM has_content and auxiliary_has_content fields for backward compatibility

- **FR-008**: System MUST update DocumentRecognitionOutput.calculate_results_status() to use preprocessing_has_content as primary input for validation logic (instead of auxiliary_has_content or VLM has_content), applying existing rules:
  - VX1 priority: If VX1.has_content is True (using heuristic, not preprocessing), results = False
  - Date fields OR: At least one of (year, month, date) must have preprocessing_has_content=True
  - Other fields AND: All non-title, non-date fields must have preprocessing_has_content=True (VX2 included with heuristic result, VX1 excluded)
  - Final results = date_valid AND other_fields_valid

- **FR-009**: System MUST update output schema to include preprocessing results:
  - JSON output: Add preprocessing_has_content and preprocessing_metrics to each field_result object
  - CSV output: Add preprocessing_{field_id}_has_content column for each field
  - Maintain existing VLM_has_content and AUX_has_content columns for comparison

- **FR-010**: System MUST update save_vlm_visualization() to use preprocessing_has_content for ROI box coloring:
  - Title fields: Original template color
  - Checkbox fields (VX1, VX2): Color based on heuristic has_content (green=True, red=False)
  - Other fields: Green (0,255,0) if preprocessing_has_content=True, Red (0,0,255) if preprocessing_has_content=False, Blue (255,0,0) if preprocessing_has_content=None (error/skip)
  - Label format: "{field_id}: {preprocessing_has_content}" or "{field_id}: ERROR" for preprocessing failures

- **FR-011**: System MUST save intermediate preprocessing images for debugging:
  - Create output/processed_rois/{document_name}/{field_id}/ subdirectory for each field
  - Save: 01_saturation.png (HSV saturation channel), 02_difference.png (template difference binary), 03_hline_removed.png (after horizontal line removal), 04_vline_removed.png (after vertical line removal), 05_noise_removed.png (after noise filtering), 06_final_binary.png (final binary image for decision)
  - Include metadata.json with preprocessing metrics (ink_ratio, component_count, has_content, thresholds used)

- **FR-012**: System MUST handle missing blank template ROIs gracefully:
  - If blank ROI image file does not exist for a field, log warning and skip preprocessing for that field
  - Fall back to VLM-only recognition (existing behavior) to maintain functionality
  - Set preprocessing_has_content to None to indicate preprocessing was skipped

- **FR-013**: System MUST implement error handling for preprocessing failures:
  - Catch image loading errors (corrupt blank template ROI)
  - Catch preprocessing errors (dimension mismatch, color space conversion failure)
  - On error: Log error with field_id and exception, set preprocessing_has_content to None, continue with VLM inference
  - Do not halt document pipeline on preprocessing failure

- **FR-014**: System MUST preserve all existing functionality when preprocessing is disabled or unavailable:
  - VLM inference runs for all fields as before
  - Existing auxiliary comparison (SIFT-based) continues to work if available
  - Checkbox heuristic recognition continues unchanged
  - Template matching and ROI extraction operate identically

- **FR-015**: System MUST log preprocessing metrics for each field for threshold tuning:
  - Log ink_ratio, component_count, and final has_content decision
  - Log reasoning: "ink_ratio=0.012 > threshold=0.005 AND component_count=5 >= threshold=3, has_content=True"
  - Include summary statistics in processing logs: "Preprocessing: 12/13 fields processed, 8 detected filled, 4 detected empty, 1 skipped"

- **FR-016**: System MUST resize sample ROI to match blank template ROI dimensions before preprocessing if sizes differ:
  - Use OpenCV resize with INTER_LINEAR interpolation
  - Preserve aspect ratio by padding with white pixels if necessary
  - Log warning if significant size mismatch (>10% difference)

### Key Entities *(include if feature involves data)*

- **BlankTemplateROI**: Represents a blank template ROI image extracted during configuration, stored at data/{template_id}/blank_rois/{field_id}.png, used as reference baseline for template difference preprocessing.

- **ROIPreprocessor**: Component responsible for executing the 6-step preprocessing pipeline on document ROI images, comparing against blank template ROIs, and producing has_content determination with explainable metrics.

- **PreprocessingResult**: Data structure containing preprocessing output for a single ROI field, including has_content decision, ink_ratio metric, component_count metric, processing_time_ms, and error information if preprocessing failed.

- **PreprocessingConfig**: Configuration parameters for the preprocessing pipeline, including all threshold values (DIFFERENCE_THRESHOLD, INK_RATIO_THRESHOLD, COMPONENT_COUNT_THRESHOLD, MIN_BLOB_AREA, MORPHOLOGY_LINE_RATIO), loaded from config.py.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Preprocessing pipeline achieves at least 98% accuracy on test documents with clear filled/unfilled fields (signature boxes, stamp areas), compared to ground truth manual labeling.

- **SC-002**: Preprocessing pipeline correctly detects faint red stamps with saturation as low as 30% (HSV S channel value 77/255), where pure grayscale approaches would fail.

- **SC-003**: Preprocessing completes in under 100ms per ROI on standard hardware (CPU-based OpenCV operations), adding minimal overhead compared to VLM inference time (typically 500-2000ms per ROI).

- **SC-004**: Document validation using preprocessing_has_content achieves at least 95% agreement with ground truth on test document set, matching or exceeding previous auxiliary comparison validation accuracy.

- **SC-005**: Saved preprocessing images enable users to debug false negatives or false positives by visually inspecting the 6-step pipeline output, with at least 80% of detection errors being explainable from the saved images.

- **SC-006**: Configuration script successfully generates blank template ROIs for all three templates (contractor_1, contractor_2, enterprise_1) in under 10 seconds, producing valid PNG files under 100KB per ROI.

- **SC-007**: Zero regression - all existing VLM recognition, output formats, checkbox detection, and auxiliary comparison features continue to function identically when preprocessing is available or unavailable.

## Assumptions

1. Blank template images provided in templates/ directory are clean scans without handwritten annotations or pre-filled content - if blank templates contain user content, template difference will incorrectly cancel out that content in sample documents.

2. Template difference preprocessing provides sufficient discrimination for typical form fields - if form design has complex overlapping patterns or textures, additional preprocessing steps (edge detection, frequency analysis) may be needed.

3. Template alignment quality is consistent between blank template ROI extraction and document ROI extraction - if alignment accuracy varies, ROI positions may not correspond correctly for template difference.

4. Preprocessing threshold parameters can be globally configured or tuned per-template - initial implementation uses single global thresholds, with future refinement allowing per-field-type or per-template thresholds.

5. Users will re-run update_configs.py whenever template definitions or ROI coordinates change - the system does not auto-detect stale blank template ROIs.

6. Preprocessing is intended for content presence detection (has_content), not content extraction - VLM handles text/number extraction, preprocessing provides binary filled/unfilled signal.

7. OpenCV morphological operations and connected components analysis are sufficient for detecting typical signature/stamp patterns - extremely unusual content patterns (single-pixel dots, very sparse content) may require custom detection logic.

8. Processing performance overhead from saving intermediate images is acceptable (estimated 20-30ms per field for 6 PNG file writes) given debugging value.

## Dependencies

- **OpenCV Python (cv2)**: Required for HSV color space conversion, morphological operations, connected components analysis, and template difference computation.

- **Blank Template Images**: Requires clean blank template source images in templates/ directory matching the configurations used for document alignment.

- **Configuration Script**: Depends on update_configs.py having access to template source images and ROI coordinate definitions from templates/location/{template_id}.json.

- **Existing ROI Extraction Pipeline**: Preprocessing pipeline assumes ROIs are already extracted via template matching and geometric correction, receiving aligned ROI images as input.

- **NumPy**: Required for pixel array operations, ratio calculations, and image manipulation.

## Out of Scope

- **Adaptive threshold tuning**: The preprocessing threshold parameters are manually configured; automatic threshold optimization based on document batch statistics is not included in this feature.

- **Machine learning-based content detection**: The preprocessing pipeline uses rule-based computer vision techniques; training a classifier on filled/unfilled ROI examples is not included.

- **Sub-ROI analysis**: The system analyzes entire ROI regions; detecting partial filling or analyzing specific sub-regions within an ROI (e.g., signature in top-left corner only) is not supported.

- **Temporal blank template updates**: If template designs change over time (new printing format), managing multiple versions of blank template ROIs per template is not handled.

- **Real-time preprocessing parameter adjustment**: Users must edit config.py and restart the application to change thresholds; live parameter tuning via UI is not provided.

- **Multi-sample blank template fusion**: If multiple blank template samples exist (different printing batches), averaging or merging them into a single reference is not implemented - only single blank ROI per field.

- **Preprocessing for checkbox fields**: Checkboxes continue using existing heuristic method; applying the 6-step preprocessing pipeline to checkboxes is explicitly out of scope to avoid regression.

- **Color-based stamp detection**: While HSV saturation helps preserve faint stamps, dedicated red/blue stamp detection using color thresholding is not implemented - preprocessing relies on template difference to isolate stamps.
