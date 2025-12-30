"""Main processing pipeline for document template alignment and ROI extraction."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from dataclasses import dataclass, field
from datetime import datetime

from vlm_pdf_recognizer.templates import GoldenTemplate
from vlm_pdf_recognizer.templates.template_loader import load_all_templates
from vlm_pdf_recognizer.preprocessing.pdf_converter import pdf_to_images
from vlm_pdf_recognizer.alignment.feature_extractor import extract_features
from vlm_pdf_recognizer.alignment.template_matcher import match_templates, UnknownDocumentError
from vlm_pdf_recognizer.alignment.geometric_corrector import align_document_to_template
from vlm_pdf_recognizer.alignment.blank_roi_cache import BlankROIFeatureCache
from vlm_pdf_recognizer.extraction.roi_extractor import extract_rois, draw_roi_boxes, ExtractedROI


@dataclass
class ProcessingResult:
    """Result of document processing pipeline."""
    input_path: str
    page_number: int
    matched_template_id: str
    confidence_score: int  # Number of RANSAC inliers
    processing_time_ms: float
    aligned_image: np.ndarray
    visualization_image: np.ndarray
    extracted_rois: List[ExtractedROI] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


class DocumentProcessor:
    """Main document processing pipeline."""

    def __init__(self, templates_dir: str = "data", verbose: bool = False):
        """
        Initialize document processor.

        Args:
            templates_dir: Directory containing template subdirectories
            verbose: Enable verbose logging
        """
        self.templates_dir = templates_dir
        self.verbose = verbose
        self.templates: List[GoldenTemplate] = []

        # Load templates
        self._load_templates()

        # Load blank ROI features cache
        self.blank_roi_cache = BlankROIFeatureCache()
        self.blank_roi_cache.load_from_directory(templates_dir)

        if self.verbose:
            loaded = len(self.blank_roi_cache.loaded_templates)
            failed = len(self.blank_roi_cache.failed_templates)
            print(f"Loaded blank ROI features: {loaded} templates")
            if failed > 0:
                print(f"  Warning: {failed} templates failed to load (using VLM-only mode)")

    def _load_templates(self):
        """Load all golden templates."""
        if self.verbose:
            print(f"Loading templates from: {self.templates_dir}")

        self.templates = load_all_templates(self.templates_dir)

        if self.verbose:
            print(f"Loaded {len(self.templates)} templates:")
            for template in self.templates:
                print(f"  - {template.template_id}: {template.feature_count()} features, {len(template.rois)} ROIs")

    def process_image(self, image: np.ndarray, page_number: int = 0,
                     input_path: str = "unknown") -> ProcessingResult:
        """
        Process a single image through the pipeline.

        Args:
            image: Input image (HxWxC, BGR, uint8)
            page_number: Page number (for multi-page PDFs)
            input_path: Original input file path

        Returns:
            ProcessingResult with aligned image, ROIs, and metadata
        """
        start_time = datetime.now()

        try:
            # Step 1: Feature extraction (use original image directly)
            # Note: SIFT is robust to watermarks, so we skip watermark removal
            # to preserve feature richness
            if self.verbose:
                print(f"  [Page {page_number}] Extracting SIFT features...")
            doc_keypoints, doc_descriptors = extract_features(
                image,  # Use original image, not preprocessed
                is_template=False
            )

            if self.verbose:
                print(f"  [Page {page_number}] Extracted {len(doc_keypoints)} features")

            # Step 3: Template matching
            if self.verbose:
                print(f"  [Page {page_number}] Matching templates...")
            match_result = match_templates(
                doc_keypoints,
                doc_descriptors,
                self.templates
            )

            if self.verbose:
                print(f"  [Page {page_number}] Matched template: {match_result.matched_template_id} "
                      f"({match_result.inlier_count} inliers)")

            # Step 4: Geometric alignment
            if self.verbose:
                print(f"  [Page {page_number}] Aligning document...")
            aligned_image = align_document_to_template(
                image,  # Use original color image
                match_result.homography_matrix,
                match_result.matched_template.image_shape
            )

            # Step 5: ROI extraction
            if self.verbose:
                print(f"  [Page {page_number}] Extracting {len(match_result.matched_template.rois)} ROIs...")
            extracted_rois = extract_rois(
                aligned_image,
                match_result.matched_template.rois
            )

            # Step 6: Visualization
            if self.verbose:
                print(f"  [Page {page_number}] Generating visualization...")
            visualization = draw_roi_boxes(aligned_image, extracted_rois)

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms

            if self.verbose:
                print(f"  [Page {page_number}] Processing completed in {processing_time:.0f}ms")

            return ProcessingResult(
                input_path=input_path,
                page_number=page_number,
                matched_template_id=match_result.matched_template_id,
                confidence_score=match_result.inlier_count,
                processing_time_ms=processing_time,
                aligned_image=aligned_image,
                visualization_image=visualization,
                extracted_rois=extracted_rois,
                success=True,
                error_message=None
            )

        except UnknownDocumentError as e:
            # Document doesn't match any template
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000

            if self.verbose:
                print(f"  [Page {page_number}] ERROR: {str(e)}")

            return ProcessingResult(
                input_path=input_path,
                page_number=page_number,
                matched_template_id="unknown",
                confidence_score=0,
                processing_time_ms=processing_time,
                aligned_image=image,
                visualization_image=image,
                extracted_rois=[],
                success=False,
                error_message=str(e)
            )

        except Exception as e:
            # Other processing errors
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000

            if self.verbose:
                print(f"  [Page {page_number}] ERROR: {str(e)}")

            return ProcessingResult(
                input_path=input_path,
                page_number=page_number,
                matched_template_id="error",
                confidence_score=0,
                processing_time_ms=processing_time,
                aligned_image=image,
                visualization_image=image,
                extracted_rois=[],
                success=False,
                error_message=str(e)
            )

    def process_file(self, input_path: str) -> List[ProcessingResult]:
        """
        Process a PDF or image file.

        Args:
            input_path: Path to PDF or image file

        Returns:
            List of ProcessingResult (one per page for PDFs)
        """
        input_path_obj = Path(input_path)

        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if self.verbose:
            print(f"\nProcessing: {input_path}")

        # Determine file type and load images
        suffix = input_path_obj.suffix.lower()

        if suffix == '.pdf':
            # PDF: convert to images
            if self.verbose:
                print("  Detected PDF, converting to images...")
            images = pdf_to_images(input_path)
            if self.verbose:
                print(f"  Loaded {len(images)} page(s)")
        elif suffix in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
            # Image: load directly
            if self.verbose:
                print("  Detected image file")
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Failed to load image: {input_path}")
            images = [image]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Process each page/image
        results = []
        for page_num, image in enumerate(images):
            result = self.process_image(image, page_num, str(input_path))
            results.append(result)

        return results

    def process_batch(self, input_paths: List[str]) -> List[ProcessingResult]:
        """
        Process multiple files in batch.

        Args:
            input_paths: List of file paths

        Returns:
            List of ProcessingResult (flattened across all files and pages)
        """
        all_results = []

        for input_path in input_paths:
            try:
                results = self.process_file(input_path)
                all_results.extend(results)
            except Exception as e:
                # Skip failed files and continue
                if self.verbose:
                    print(f"ERROR processing {input_path}: {str(e)}")
                # Create error result
                error_result = ProcessingResult(
                    input_path=input_path,
                    page_number=0,
                    matched_template_id="error",
                    confidence_score=0,
                    processing_time_ms=0,
                    aligned_image=np.zeros((100, 100, 3), dtype=np.uint8),
                    visualization_image=np.zeros((100, 100, 3), dtype=np.uint8),
                    extracted_rois=[],
                    success=False,
                    error_message=str(e)
                )
                all_results.append(error_result)

        return all_results
