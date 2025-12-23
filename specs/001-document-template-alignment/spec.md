# Feature Specification: Document Template Alignment & ROI Extraction

**Feature Branch**: `001-document-template-alignment`
**Created**: 2025-12-23
**Status**: Draft
**Input**: User description: "Local, zero-shot document processing system for Traditional Chinese scanned PDF documents with template-based classification, alignment, and ROI extraction"

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Process Single Scanned Document (Priority: P1)

A user has a scanned PDF document (enterprise or contractor type) with potential watermarks, skew, and handwritten annotations. They need the system to automatically identify which template it matches, remove visual noise, align it to the standard template, and extract specific regions of interest.

**Why this priority**: This is the core use case - without this, the system cannot deliver any value. All other features depend on this working correctly.

**Independent Test**: Can be fully tested by providing a single scanned document as input and verifying that (1) correct template is identified, (2) document is geometrically aligned, (3) ROI bounding boxes are correctly overlaid on output image, and delivers a validated, aligned document with marked ROIs.

**Acceptance Scenarios**:

1. **Given** a scanned contractor document with blue watermarks and 5-degree rotation, **When** user provides the document to the system, **Then** system identifies it as contractor_1 or contractor_2 template, removes watermarks, corrects rotation, and outputs aligned image with ROI boundaries drawn
2. **Given** a scanned enterprise document with handwritten notes and stamps, **When** user processes the document, **Then** system preserves black/dark blue pen marks and red stamps while removing colored backgrounds, aligns to enterprise_1 template, and marks all configured ROI regions
3. **Given** a clean scanned document with minimal skew, **When** user processes the document, **Then** system completes alignment in under 10 seconds and outputs verification image showing matched template and ROI boxes

---

### User Story 2 - Handle Unknown or Poor Quality Documents (Priority: P2)

A user attempts to process a document that either doesn't match any known template or has such poor quality that feature matching fails. The system needs to clearly indicate failure rather than producing incorrect results.

**Why this priority**: Error handling prevents misleading outputs and saves user time by immediately flagging problematic inputs rather than producing unreliable results.

**Independent Test**: Can be fully tested by providing (1) a document that matches none of the three templates, (2) a severely blurred document, and (3) a document with fewer than 50 good feature matches, then verifying appropriate error messages are returned for each case.

**Acceptance Scenarios**:

1. **Given** a completely different document type (e.g., passport scan), **When** user attempts to process it, **Then** system returns "Unknown Document" error with message indicating no template matched
2. **Given** a heavily corrupted or blurred scan with insufficient feature points, **When** user processes the document, **Then** system returns error stating "Insufficient feature matches (found X, need 50+)"
3. **Given** a document that partially matches multiple templates with similar confidence, **When** user processes it, **Then** system either selects the highest-confidence match or returns ambiguity warning with match scores for each template

---

### User Story 3 - Batch Process Multiple Documents (Priority: P3)

A user has a folder containing multiple scanned documents of different types (mix of enterprise and contractor forms). They want to process all documents in one operation with the system automatically routing each to its correct template.

**Why this priority**: Improves efficiency for users with many documents, but depends on P1 working reliably first. This is a convenience enhancement rather than core functionality.

**Independent Test**: Can be fully tested by creating a folder with 10 mixed documents (3 enterprise_1, 4 contractor_1, 3 contractor_2), processing them all, and verifying each is correctly classified and aligned to its respective template with outputs saved to organized subdirectories.

**Acceptance Scenarios**:

1. **Given** a directory containing 20 mixed document types, **When** user initiates batch processing, **Then** system processes all documents, outputs aligned images with ROI boxes to organized folders by template type, and generates summary report showing count per template type
2. **Given** a batch where 3 out of 15 documents fail matching, **When** batch processing completes, **Then** system successfully processes 12 documents, moves 3 failed documents to error folder, and provides detailed error log listing why each failed
3. **Given** a batch processing job in progress, **When** user monitors the process, **Then** system displays progress indicator showing current document number and template classification results

---

### Edge Cases

- What happens when a document is upside-down or rotated 180 degrees?
- How does system handle documents photographed instead of scanned (with perspective distortion)?
- What if a document has extreme skew (>30 degrees rotation)?
- How does system handle partially cropped documents missing sections?
- What if watermark colors overlap with actual content (e.g., blue ink pen on blue watermark)?
- How does system behave when template images themselves are corrupted or missing?
- What if JSON configuration file has invalid or out-of-bounds ROI coordinates?
- How does system handle multi-page PDFs versus single-page images?
- What happens when processing on CPU versus GPU - are results identical?
- How does system perform when available RAM is insufficient for high-resolution scans?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support three document template types: enterprise_1, contractor_1, and contractor_2
- **FR-002**: System MUST load golden template images and their pre-computed SIFT features from data/ directory at initialization
- **FR-003**: System MUST load ROI coordinate configurations from JSON files in data/ directory, where coordinates define bounding boxes (top-left to bottom-right)
- **FR-004**: System MUST perform watermark removal using HSV color space thresholding to eliminate blue, light gray, and light red watermarks
- **FR-005**: System MUST preserve black text, table lines, dark blue pen marks, and red stamps during watermark removal
- **FR-006**: System MUST convert preprocessed images to pure binary (black and white, 0 and 255 only)
- **FR-007**: System MUST extract SIFT feature points and descriptors from input documents
- **FR-008**: System MUST match input document features against all three template types using FLANN-based matcher or BFMatcher
- **FR-009**: System MUST identify document type based on voting mechanism - template with most good matches (inliers) wins
- **FR-010**: System MUST compute Homography matrix using RANSAC algorithm to eliminate noise from handwriting and stamps
- **FR-011**: System MUST apply perspective warp transformation to align input document to template coordinate system
- **FR-012**: System MUST extract ROI regions from aligned document based on template-specific JSON coordinates
- **FR-013**: System MUST overlay ROI bounding boxes on aligned output image for verification purposes
- **FR-014**: System MUST save verification images to output/ directory
- **FR-015**: System MUST cache computed SIFT features and Homography matrices for the three golden templates to avoid recomputation
- **FR-016**: System MUST throw "Unknown Document" exception when number of good feature matches is below 50
- **FR-017**: System MUST automatically detect and utilize GPU acceleration when available (with 4GB VRAM minimum), falling back to CPU when GPU is unavailable
- **FR-018**: System MUST operate in vlmcv Python environment
- **FR-019**: System MUST process scanned PDF documents containing Traditional Chinese text
- **FR-020**: System MUST handle document skew, rotation, and positional offset through geometric correction
- **FR-021**: System MUST accept both PDF files and image files (PNG, JPEG) as input
- **FR-022**: System MUST convert each page of a multi-page PDF to a separate image, preserving original page dimensions
- **FR-023**: System MUST process each PDF page independently, producing separate aligned outputs with ROI overlays for each page
- **FR-024**: System MUST skip and continue processing remaining documents when a document fails in batch mode, logging errors for review

### Key Entities

- **Golden Template**: Reference document image for each template type (enterprise_1, contractor_1, contractor_2) with pre-computed SIFT features stored in data/ directory
- **Template Configuration**: JSON file containing template-specific metadata including ROI bounding box coordinates (top-left to bottom-right format)
- **Input Document**: Scanned PDF or image file potentially containing watermarks, skew, handwritten annotations, and stamps
- **ROI (Region of Interest)**: Specific rectangular area on template defined by bounding box coordinates, representing data field to be extracted
- **Feature Match**: Pair of corresponding SIFT keypoints between input document and template, with quality metric (inlier/outlier)
- **Homography Matrix**: 3x3 transformation matrix mapping input document coordinates to template coordinate system
- **Verification Output**: Aligned document image with ROI bounding boxes overlaid, saved to output/ directory for validation

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System correctly identifies document template type with 95% accuracy for documents with standard quality scans
- **SC-002**: System processes single document from input to verified output in under 10 seconds on CPU-only environment
- **SC-003**: System processes single document in under 3 seconds when GPU acceleration is available
- **SC-004**: Watermark removal preserves 100% of black text and table lines while eliminating at least 90% of colored watermark pixels
- **SC-005**: Geometric alignment achieves pixel-level accuracy within 5 pixels RMSE (Root Mean Square Error) when compared to golden template
- **SC-006**: System successfully processes documents with skew up to 15 degrees without manual intervention
- **SC-007**: ROI extraction accuracy achieves 98% correct boundary detection when validated against ground truth annotations
- **SC-008**: System operates on hardware with as little as 4GB VRAM (GPU) or 8GB RAM (CPU-only mode)
- **SC-009**: Cached template features reduce subsequent processing time by at least 40% compared to computing features on every run
- **SC-010**: Error detection correctly rejects 100% of documents that don't match any template (no false positives)
- **SC-011**: System processes batch of 100 documents with less than 3% failure rate for standard quality scans

## Assumptions

1. **Input Format**: System accepts both PDF files and image formats (PNG, JPEG). Multi-page PDFs are processed with each page converted to a separate image, preserving original dimensions.
2. **Template Stability**: Assuming the three template types (enterprise_1, contractor_1, contractor_2) are fixed and won't change frequently. New template addition is not in scope.
3. **Language Processing**: System focuses on visual processing (alignment and extraction) only. OCR or text recognition of Traditional Chinese characters is explicitly out of scope for this phase.
4. **Watermark Colors**: Assuming watermarks are primarily blue, light gray, or light red. Other watermark colors may not be effectively removed.
5. **Document Quality**: Assuming scanned documents are at least 150 DPI resolution. Lower resolution may cause feature matching failures.
6. **Hardware Access**: Assuming system has exclusive access to GPU when available (no resource contention with other applications).
7. **Golden Template Quality**: Assuming golden template images are high-quality, clean scans without any defects or artifacts.
8. **ROI Coordinates**: Assuming JSON configuration files are manually created and validated before system deployment. No runtime validation or coordinate correction is performed.
9. **File System Access**: Assuming system has read access to data/ directory and write access to output/ directory.
10. **Python Environment**: Assuming vlmcv environment is pre-configured with necessary OpenCV and vision processing libraries.

## Out of Scope

- OCR (Optical Character Recognition) or text extraction from ROI regions
- Training custom models or fine-tuning existing models
- Support for document types beyond the three predefined templates
- Real-time video stream processing
- Automatic template creation or learning from new document types
- Cloud-based processing or API deployment
- User interface or web application (command-line/script execution only)
- Handling color documents that are not primarily black text on white/colored background
- Processing documents in languages other than Traditional Chinese (though system is language-agnostic for visual processing)
- Data privacy compliance features (encryption, anonymization)
- Document classification beyond template matching (e.g., semantic content analysis)

## Dependencies

1. **OpenCV Library**: Core dependency for all image processing, feature detection (SIFT), feature matching, and geometric transformations
2. **PDF Processing Library**: Library for converting PDF pages to images (e.g., pdf2image, PyMuPDF) while preserving original dimensions
3. **Python Environment (vlmcv)**: Pre-configured environment with vision and machine learning libraries
4. **GPU Drivers**: CUDA-compatible drivers if GPU acceleration is to be utilized (optional, system must work without)
5. **Golden Template Assets**: Three clean template images must exist in data/ directory before system initialization
6. **Configuration Files**: Three JSON files (one per template) defining ROI coordinates must exist in data/ directory
7. **File System**: Read/write permissions for data/ and output/ directories
8. **Compute Resources**: Minimum 4GB VRAM (GPU mode) or 8GB RAM (CPU mode)

## Clarifications

**PDF Handling**: System accepts both image files and PDF files. For PDF inputs, system converts each page to a separate image while preserving original dimensions. Multi-page PDFs result in multiple processed outputs (one per page). Template images are provided as image files.

**Template Match Ambiguity**: System always uses highest-scoring template (winner-takes-all approach) regardless of confidence margin between templates.

**Batch Processing Error Handling**: Failed documents are skipped, batch processing continues with remaining documents, and errors are logged for later review.
