# Feature Specification: VLM Auxiliary ROI Comparison System

**Feature Branch**: `003-vlm-auxiliary-roi-comparison`
**Created**: 2025-12-30
**Status**: Draft
**Input**: User description: "VLM輔助辨識功能
1. 詳細閱讀現在所有的程式碼和文件,不能遺失或修改錯誤現有的運作功能和邏輯。
2. 新功能要完美整合銜接到現有的程式碼中,不能有錯誤或冗餘。
3. update_configs.py除了現有功能是產生template的向量提供對齊之外,還要根據template/提供的座標檔進行才切出ROI後產個中template的ROI相量儲存於data(空白樣本roi向量)。
4. 在辨識新樣本要將ROI放入VLM辨識時,先把新樣本的ROi跟空白樣本的ROi做比對(類似第一階段樣本對齊辨識template方法),諾新樣本ROI跟空白樣本ROI有很高的相似度,則代表該樣本沒有填寫蓋章或簽名。如果有差異代表可能有填寫蓋章或簽名,再送入VLM辨識。
5. 輸出維持現有的VLM辨識所有ROi的輸出,新增一欄輔助辨識的比對結果。
6. 輸出邏輯維持現有一樣,只是將result的判斷改為輔助辨識的比對結果優先,因為result目前只看有無不看內容,邏輯要跟現在一樣(現有邏輯是:VX1 True則result直接True,其餘所有內容應該都要是True,如果有False則result是False(除了year, month, date三個用or判斷如目前的邏輯,請先確認目前邏輯))
7. 視覺化的輸出結果也是改成用輔助辨識的結果True or False呈現紅色和綠色
8. VX1, VX2辨識checkbox維持現有方式"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Blank Template ROI Feature Extraction (Priority: P1)

During system configuration, administrators extract blank template images and generate reference ROI features for each field position. These blank ROI features serve as baseline comparisons to distinguish unfilled fields from completed ones during document processing.

**Why this priority**: Foundational capability - without blank template ROI features, the auxiliary comparison system cannot function. This establishes the reference data needed for all subsequent validation.

**Independent Test**: Run the updated configuration script on template images, verify that blank ROI images are extracted to data/{template_id}/rois/ directory, and SIFT feature vectors are saved to data/{template_id}/blank_roi_features.npz for each template.

**Acceptance Scenarios**:

1. **Given** a blank contractor_1 template with 13 defined ROI coordinates, **When** the configuration script runs, **Then** 13 blank ROI images are extracted and saved with their field IDs, and 13 sets of SIFT features are stored in the feature cache file.

2. **Given** a blank enterprise_1 template with 14 ROI fields, **When** feature extraction completes, **Then** the feature cache contains keypoints and descriptors for all 14 fields indexed by field_id.

3. **Given** an updated template configuration with modified ROI coordinates, **When** the configuration script re-runs, **Then** existing blank ROI features are replaced with newly extracted features matching the updated coordinates.

---

### User Story 2 - Auxiliary ROI Comparison Before VLM Inference (Priority: P1)

When processing a new document, the system compares each extracted ROI against its corresponding blank template ROI using feature matching. High similarity indicates an unfilled field (user did not add content), while low similarity indicates potential content presence (signature, stamp, or text filled in).

**Why this priority**: Core differentiation - this auxiliary check prevents unnecessary VLM calls on clearly empty fields and provides a more reliable content presence indicator than VLM alone for visual elements like stamps and signatures.

**Independent Test**: Process a document with mixed filled/unfilled fields, verify that the auxiliary_has_content column in output shows correct True/False based on ROI similarity comparison (independent of VLM results), with similarity scores logged for debugging.

**Acceptance Scenarios**:

1. **Given** a document ROI that is visually identical to the blank template (no signature/stamp added), **When** SIFT feature matching computes similarity, **Then** the inlier count is above the threshold (e.g., 80% of blank template features matched), and auxiliary_has_content is set to False.

2. **Given** a document ROI with a stamped seal covering 30% of the area, **When** feature matching runs, **Then** the inlier count drops significantly (e.g., <50% match), and auxiliary_has_content is set to True, triggering VLM inference.

3. **Given** a document ROI with handwritten signature, **When** comparison against blank ROI executes, **Then** feature matching detects low similarity, auxiliary_has_content is True, and the ROI proceeds to VLM recognition.

---

### User Story 3 - Integrated Output with Auxiliary Comparison Results (Priority: P1)

The system outputs recognition results containing both VLM inference outcomes and auxiliary comparison results for each field. Users can review both data sources to understand how each field was validated.

**Why this priority**: Transparency requirement - users need visibility into both the pixel-based comparison and VLM inference to trust validation results and debug edge cases where methods disagree.

**Independent Test**: Process a test document and verify that the output CSV/JSON contains all existing VLM columns plus new auxiliary_has_content column for each field, with values correctly populated from the ROI comparison logic.

**Acceptance Scenarios**:

1. **Given** a processed document with 13 ROI fields, **When** output generation completes, **Then** the result file contains 13 rows with both vlm_has_content and auxiliary_has_content columns populated.

2. **Given** a field where VLM detected content but auxiliary comparison shows high similarity to blank, **When** viewing output, **Then** both values are visible (e.g., vlm_has_content=True, auxiliary_has_content=False), allowing downstream analysis.

3. **Given** a title field that skips auxiliary comparison, **When** output is generated, **Then** auxiliary_has_content is set to None or excluded for title fields, maintaining consistent schema.

---

### User Story 4 - Auxiliary-Priority Result Validation Logic (Priority: P1)

The system calculates the overall document validation status (results field) by prioritizing auxiliary comparison results over VLM inference for has_content determination. The existing validation logic (VX1 priority, date field OR, other fields AND) is preserved but uses auxiliary_has_content as the primary input.

**Why this priority**: Accuracy improvement - auxiliary comparison is more reliable for visual content detection (stamps, signatures) than VLM zero-shot inference, reducing false positives from VLM hallucination.

**Independent Test**: Process documents with known completion status, verify that the results field correctly reflects validation using auxiliary_has_content values, matching the existing logic rules (VX1 checked = False, date fields use OR, other fields use AND).

**Acceptance Scenarios**:

1. **Given** a document where VX1 auxiliary_has_content is True (checkbox marked), **When** validation logic runs, **Then** results is immediately set to False regardless of other field values.

2. **Given** a document where all date fields (year, month, date) have auxiliary_has_content=False, **When** validation calculates results, **Then** results is False because the date OR condition fails.

3. **Given** a document where at least one date field has auxiliary_has_content=True and all other non-title fields have auxiliary_has_content=True, **When** validation runs, **Then** results is True.

4. **Given** a document where one required field (e.g., company1) has auxiliary_has_content=False, **When** validation logic executes, **Then** results is False because the AND condition for other fields fails.

---

### User Story 5 - Color-Coded Visualization with Auxiliary Results (Priority: P2)

The system generates visualization images with ROI bounding boxes colored based on auxiliary_has_content values (green for True, red for False), replacing the previous VLM-based coloring. This provides visual quick-scan of document completion status.

**Why this priority**: User experience enhancement - visual feedback is faster than reading CSV files for batch validation, and auxiliary results are more trustworthy for visual inspection.

**Independent Test**: Process a document and verify that the visualization image shows green boxes around filled fields (high ROI difference) and red boxes around unfilled fields (high ROI similarity to blank template).

**Acceptance Scenarios**:

1. **Given** a document with 10 completed fields and 3 empty fields, **When** visualization is generated, **Then** 10 ROI boxes are drawn in green (auxiliary_has_content=True) and 3 boxes in red (auxiliary_has_content=False).

2. **Given** a field where auxiliary comparison failed to run (error case), **When** visualization renders, **Then** the ROI box defaults to the original template color or a neutral color (e.g., blue) to indicate uncertainty.

3. **Given** a title field that has no auxiliary comparison, **When** visualization is drawn, **Then** the title ROI box uses the original template color (typically green) since it's not subject to validation.

---

### User Story 6 - Checkbox Recognition Preservation (Priority: P1)

VX1 and VX2 checkbox fields continue to use the existing hybrid recognition method (VLM + heuristic pixel check) without auxiliary ROI comparison, as their current implementation already provides reliable detection.

**Why this priority**: Regression prevention - checkbox recognition is working well with the current heuristic approach; introducing auxiliary comparison could reduce accuracy for small checkmark variations.

**Independent Test**: Process documents with checked/unchecked VX1 and VX2 boxes, verify that recognition results match the existing behavior (heuristic-primary with VLM fallback), and auxiliary_has_content is not used for checkbox validation logic.

**Acceptance Scenarios**:

1. **Given** a document with VX1 checkbox checked, **When** recognition runs, **Then** the existing heuristic detects the checkmark, auxiliary comparison is skipped for VX1, and VX1.has_content reflects heuristic result.

2. **Given** a document with VX2 checkbox unchecked, **When** recognition executes, **Then** heuristic returns False, auxiliary comparison is bypassed, and VX2.has_content is False.

3. **Given** checkbox field results, **When** validation logic runs, **Then** VX1 and VX2 use their existing has_content values (from heuristic) rather than auxiliary_has_content for the results calculation.

---

### Edge Cases

- **Blank template with printing variations**: What happens when the blank template has slight printing defects or quality variations? The SIFT feature matching should be robust to minor noise; if excessive false positives occur, adjust the similarity threshold or use multiple blank template samples.

- **Partial content filling**: How does the system handle fields with very light signatures or faint stamps? Auxiliary comparison will detect difference from blank (low similarity), correctly marking auxiliary_has_content=True, then VLM attempts extraction.

- **Template alignment errors**: What happens when document alignment fails and ROI positions are incorrect? Auxiliary comparison will likely fail to match blank template features (wrong region extracted); system should detect low match quality and fall back to VLM-only recognition with logged warning.

- **Missing blank ROI features**: How does system behave if blank_roi_features.npz is not found for a template? System should log error, skip auxiliary comparison for that template, and fall back to existing VLM-only recognition to maintain functionality.

- **New templates without blank features**: What happens when processing documents with newly added templates that lack blank ROI feature files? Gracefully fall back to VLM-only mode with warning log; users should run update_configs.py to generate missing blank features.

- **ROI size mismatches**: What happens if the blank template ROI and document ROI have different dimensions? Resize the document ROI to match blank template ROI dimensions before feature extraction to ensure comparable feature sets.

- **Multi-page PDFs with mixed templates**: How does system select correct blank ROI features for each page? Match based on the template_id assigned during alignment phase; each page's ROIs compare against its matched template's blank features.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extend update_configs.py to extract blank ROI images from template source images using the configured ROI coordinates, saving ROI images to data/{template_id}/rois/blank_{field_id}.png.

- **FR-002**: System MUST extract SIFT feature descriptors (keypoints and descriptors) from each blank template ROI image and store them in a feature cache file at data/{template_id}/blank_roi_features.npz with field_id as the key for each feature set.

- **FR-003**: System MUST load blank ROI features during processor initialization and cache them in memory for runtime comparison, validating that all expected field_ids have corresponding blank features before processing documents.

- **FR-004**: System MUST implement an ROI comparison function that accepts a document ROI image and its corresponding blank template ROI features, performing SIFT feature matching with the following steps:
  - Extract SIFT features from document ROI
  - Match document ROI descriptors against blank ROI descriptors using FLANN-based matcher
  - Compute inlier count using RANSAC homography estimation (similar to template matching logic)
  - Calculate similarity ratio as inlier_count / min(doc_features, blank_features)
  - Return similarity score and auxiliary_has_content boolean (True if similarity < threshold, False if similarity >= threshold)

- **FR-005**: System MUST apply auxiliary ROI comparison before VLM inference for all non-title, non-checkbox fields (text, number, stamp fields), using a similarity ratio threshold of 0.6 (60% feature match) - fields with ≥60% feature similarity to blank template are marked as unfilled (auxiliary_has_content=False), fields with <60% similarity are marked as potentially filled (auxiliary_has_content=True)

- **FR-006**: System MUST skip auxiliary ROI comparison for title fields (no validation needed) and checkbox fields (existing heuristic is sufficient), setting auxiliary_has_content to None for these field types.

- **FR-007**: System MUST conditionally invoke VLM inference based on auxiliary comparison results:
  - If auxiliary_has_content is False (high similarity to blank), skip VLM inference and set vlm_has_content to None (no content detected, VLM not needed)
  - If auxiliary_has_content is True (low similarity, potential content), proceed with VLM inference as usual
  - If auxiliary comparison failed or was skipped, always run VLM inference (fallback to existing behavior)

- **FR-008**: System MUST store both auxiliary_has_content and vlm_has_content values in the RecognitionResult dataclass for each field, preserving both data points for output and debugging.

- **FR-009**: System MUST update the DocumentRecognitionOutput.calculate_results_status() method to use auxiliary_has_content as the primary input for validation logic (instead of vlm_has_content), applying the existing rules:
  - VX1 priority: If VX1.has_content is True (using existing heuristic result, not auxiliary), results = False immediately
  - Date fields OR: At least one of (year, month, date) must have auxiliary_has_content=True
  - Other fields AND: All non-title, non-date fields must have auxiliary_has_content=True (VX2 included, VX1 excluded)
  - Final results = date_valid AND other_fields_valid

- **FR-010**: System MUST update output schema to include auxiliary comparison results:
  - JSON output: Add auxiliary_has_content field to each field_result object in the fields dictionary
  - CSV output: Add auxiliary_{field_id}_has_content column for each field (parallel to existing {field_id}_has_content columns from VLM)
  - Maintain all existing VLM output columns for backward compatibility

- **FR-011**: System MUST update save_vlm_visualization() function to use auxiliary_has_content for ROI box coloring (instead of vlm_has_content):
  - Title fields: Use original template color (typically green)
  - Checkbox fields (VX1, VX2): Use existing heuristic has_content for color coding
  - Other fields: Green (0,255,0) if auxiliary_has_content=True, Red (0,0,255) if auxiliary_has_content=False
  - Label format: "{field_id}: {auxiliary_has_content}" (e.g., "company1: True")

- **FR-012**: System MUST handle missing blank ROI features gracefully:
  - If blank_roi_features.npz does not exist for a template, log warning and skip auxiliary comparison for all fields in that template
  - Fall back to VLM-only recognition (existing behavior) to maintain functionality
  - Document processor should not crash or halt due to missing blank features

- **FR-013**: System MUST implement error handling for auxiliary comparison failures:
  - Catch feature extraction errors (insufficient keypoints, descriptor computation failures)
  - Catch matching errors (FLANN matcher exceptions, homography estimation failures)
  - On error: Log warning, set auxiliary_has_content to None, proceed with VLM inference
  - Continue processing other ROIs without halting the document pipeline

- **FR-014**: System MUST preserve all existing functionality when auxiliary comparison is disabled or unavailable:
  - VLM inference still runs for all fields (except those skipped by successful auxiliary detection)
  - Existing output formats remain valid (with auxiliary columns as additional data)
  - Checkbox heuristic recognition continues unchanged
  - Template matching and ROI extraction operate identically

- **FR-015**: System MUST log auxiliary comparison metrics for debugging and threshold tuning:
  - Log similarity scores and inlier counts for each ROI comparison
  - Log the final auxiliary_has_content decision with reasoning (e.g., "similarity=0.82 >= threshold=0.70, auxiliary_has_content=False")
  - Include comparison statistics in processing logs (e.g., "Auxiliary comparison: 10/13 fields compared, 7 detected as filled, 3 as empty")

### Key Entities *(include if feature involves data)*

- **BlankROIFeatureCache**: Data structure storing SIFT features (keypoints, descriptors) for all blank template ROI regions, indexed by template_id and field_id, loaded from data/{template_id}/blank_roi_features.npz.

- **ROIComparator**: Component responsible for comparing document ROI features against blank template ROI features, calculating similarity scores, and determining auxiliary_has_content values.

- **AuxiliaryRecognitionResult**: Extended recognition result containing both auxiliary_has_content (from pixel-based comparison) and vlm_has_content (from VLM inference), plus similarity_score metadata.

- **BlankTemplateROI**: Represents a blank template ROI image and its extracted SIFT features, used as reference baseline for document ROI comparison.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Auxiliary ROI comparison reduces unnecessary VLM inference calls by at least 30% on typical document batches (fields correctly identified as empty without VLM), improving processing throughput.

- **SC-002**: Overall has_content detection accuracy (combining auxiliary comparison + VLM) improves to at least 97% on test documents with clear filled/unfilled fields, compared to 95% with VLM-only.

- **SC-003**: Auxiliary comparison completes in under 50ms per ROI on standard hardware (comparable to template matching performance), adding negligible overhead to the pipeline.

- **SC-004**: Document validation results using auxiliary-priority logic match or exceed the accuracy of VLM-only validation, with at least 95% agreement on known-status test documents.

- **SC-005**: Zero regression - all existing VLM recognition, output formats, and checkbox detection continue to function identically when auxiliary comparison is disabled or unavailable.

- **SC-006**: Configuration script successfully generates blank ROI features for all three templates (contractor_1, contractor_2, enterprise_1) without errors, producing valid feature cache files under 5MB per template.

## Assumptions

1. Blank template images provided in templates/ directory are high-quality scans without handwritten annotations or stamps - if blank templates contain pre-filled content, auxiliary comparison will incorrectly identify filled fields as empty.

2. SIFT feature matching provides sufficient discrimination between blank and filled ROIs - if field content is too sparse (e.g., single small checkmark in large text field), features may not differ enough from blank template.

3. Template alignment quality is consistent between blank template feature extraction and document processing - if alignment accuracy varies, ROI positions may not correspond correctly.

4. The similarity threshold for auxiliary_has_content can be tuned globally or per-field-type - initial implementation uses a single threshold, with future refinement based on real-world performance data.

5. Users will re-run update_configs.py whenever template ROI coordinates change - the system does not auto-detect stale blank features and regenerate them.

6. Blank template ROI features are static and do not change during runtime - feature cache files are loaded once at initialization and reused for all documents.

7. Auxiliary comparison is intended for content presence detection, not content extraction - it provides a binary filled/unfilled signal, while VLM handles text/number extraction when content is detected.

8. Processing performance overhead from auxiliary comparison is acceptable (estimated 10-20% increase in total pipeline time) given the accuracy and efficiency benefits.

## Dependencies

- **Existing Template Matching Logic**: Auxiliary ROI comparison reuses the SIFT feature extraction and FLANN-based matching infrastructure from vlm_pdf_recognizer.alignment modules.

- **Blank Template Images**: Requires clean, unfilled template images in templates/ directory that match the configurations used for document alignment.

- **Configuration Script**: Depends on update_configs.py having access to template source images and ROI coordinate definitions from templates/location/{template_id}.json.

- **SIFT Feature Compatibility**: Assumes SIFT features extracted from blank ROIs are comparable to features extracted from aligned document ROIs (same OpenCV version, same preprocessing steps).

## Out of Scope

- **Multi-template blank feature fusion**: If multiple blank template samples exist (different printing batches), averaging or merging their features is not implemented - only single blank template per template_id.

- **Adaptive threshold tuning**: The similarity threshold for auxiliary_has_content is manually configured; automatic threshold optimization based on document batch statistics is not included.

- **Auxiliary comparison for checkboxes**: Checkbox fields continue using the existing heuristic method; auxiliary ROI comparison is not applied to VX1/VX2 despite potential benefits.

- **Sub-ROI analysis**: The system compares entire ROI regions; detecting partial filling or analyzing specific sub-regions within an ROI (e.g., signature in top-left corner only) is not supported.

- **Temporal blank template updates**: If template designs change over time (new printing format), managing multiple versions of blank features per template is not handled.

- **Real-time feedback**: The system processes documents in batch mode; interactive preview showing auxiliary comparison results before VLM inference is not provided.
