"""
ROI Preprocessing Pipeline - Cleaned Version.

This module implements a simple BGR difference-based preprocessing pipeline to detect
user content in ROI fields.

Pipeline Strategy:
    1. ECC Alignment: Fine-grained ROI-level alignment to template
    2. BGR Difference: Direct absolute difference on all channels
    3. Multi-stage Detection: Filter pre-printed text residuals
    4. Decision: Mean difference threshold with significant pixel ratio

This simplified approach focuses on direct color difference rather than complex
morphological operations or density maps.
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict

import cv2
import numpy as np

# ==============================================================================
# ROI PREPROCESSING PARAMETERS
# ==============================================================================
# Minimum mean difference threshold for normal fields (no pre-printed text)
MIN_ABSOLUTE_DENSITY_THRESHOLD = 0.01
# PNG compression level for debug images (0=fast/large, 9=slow/small)
PNG_COMPRESSION_LEVEL = 6
# Pre-allocate work buffers for preprocessing pipeline
PRE_ALLOCATE_WORK_BUFFERS = True
# Maximum ROI dimension (width or height) for buffer pre-allocation
MAX_ROI_DIMENSION = 1000


logger = logging.getLogger(__name__)


@dataclass
class AIPResult:
    """Output of AIP (Advanced Image Processing) pipeline for a single ROI."""
    field_id: str
    has_content: Optional[bool]  # True=content, False=empty, None=error
    ink_ratio: Optional[float]  # Mean difference score (0.0-1.0), None if error
    component_count: Optional[int]  # Not used, kept for compatibility
    processing_time_ms: float  # Always populated
    error_message: Optional[str] = None  # None if successful
    processed_image: Optional[np.ndarray] = None  # Difference image for visualization


class ROIPreprocessor:
    """
    Simple BGR difference-based AIP (Advanced Image Processing) pipeline for ROI content detection.

    Pipeline Strategy:
        1. ECC Alignment: Align document ROI to template ROI
        2. BGR Difference: Compute absolute difference across all channels
        3. Multi-stage Detection: Filter pre-printed text residuals
        4. Decision: Mean difference threshold with significant pixel ratio

    Thread Safety:
        NOT thread-safe - reuses internal work buffers
        Create one instance per thread for parallel processing
    """

    def __init__(
        self,
        save_debug_images: bool = False,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize ROI AIP (Advanced Image Processing) pipeline.

        Args:
            save_debug_images: Whether to save intermediate images for debugging
            output_dir: Directory path for saving debug images (required if save_debug_images=True)

        Raises:
            ValueError: If save_debug_images=True but output_dir is None
        """
        self.save_debug_images = save_debug_images
        self.output_dir = Path(output_dir) if output_dir else None

        if save_debug_images and output_dir is None:
            raise ValueError("output_dir must be provided when save_debug_images=True")

        # Create output directory if needed
        if self.save_debug_images and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-allocate work buffers if configured
        self.work_buffers = {}
        if PRE_ALLOCATE_WORK_BUFFERS:
            max_dim = MAX_ROI_DIMENSION
            self.work_buffers['gray'] = np.zeros((max_dim, max_dim), dtype=np.uint8)
            self.work_buffers['binary'] = np.zeros((max_dim, max_dim), dtype=np.uint8)
            self.work_buffers['float'] = np.zeros((max_dim, max_dim), dtype=np.float32)

    def preprocess_roi(
        self,
        doc_roi_image: np.ndarray,
        blank_template_roi: np.ndarray,
        field_id: str,
        document_name: Optional[str] = None
    ) -> AIPResult:
        """
        Execute BGR difference-based AIP (Advanced Image Processing) pipeline on document ROI.

        Args:
            doc_roi_image: Document ROI image, BGR format, uint8, shape (H, W, 3)
            blank_template_roi: Blank template ROI image, BGR format, uint8, shape (H, W, 3)
            field_id: Field identifier for logging and debug image naming
            document_name: Document name for debug image directory structure (optional)

        Returns:
            AIPResult with has_content determination and mean_diff score

        Raises:
            ValueError: If doc_roi_image or blank_template_roi are invalid (wrong shape, dtype)
        """
        start_time = time.time()

        try:
            # Validate input images
            if doc_roi_image.dtype != np.uint8 or len(doc_roi_image.shape) != 3:
                raise ValueError(
                    f"Invalid doc_roi_image format: dtype={doc_roi_image.dtype}, shape={doc_roi_image.shape}"
                )
            if blank_template_roi.dtype != np.uint8 or len(blank_template_roi.shape) != 3:
                raise ValueError(
                    f"Invalid blank_template_roi format: dtype={blank_template_roi.dtype}, shape={blank_template_roi.shape}"
                )

            # Handle dimension mismatch
            if doc_roi_image.shape != blank_template_roi.shape:
                doc_h, doc_w = doc_roi_image.shape[:2]
                tpl_h, tpl_w = blank_template_roi.shape[:2]

                size_diff = abs(doc_w - tpl_w) / tpl_w + abs(doc_h - tpl_h) / tpl_h
                if size_diff > 0.1:  # More than 10% difference
                    logger.warning(
                        f"{field_id}: ROI size mismatch >10%: doc={doc_w}x{doc_h}, template={tpl_w}x{tpl_h}"
                    )

                # Resize doc ROI to match template
                doc_roi_image = cv2.resize(doc_roi_image, (tpl_w, tpl_h))

            # Store intermediate images for debugging
            intermediate_images = {}

            # Step 1: ROI-level alignment for fields with pre-printed text
            # Use ECC to align doc ROI with blank template ROI
            # This ensures pre-printed text perfectly overlaps before difference
            aligned_doc_roi = self._align_roi_to_template(doc_roi_image, blank_template_roi)
            intermediate_images['00_aligned_doc'] = aligned_doc_roi

            # Step 2: Direct BGR difference (simplest approach)
            # Both document and template are in original BGR format
            # Compute absolute difference to detect user-added content

            # Convert to float for computation
            doc_float = aligned_doc_roi.astype(np.float32)
            template_float = blank_template_roi.astype(np.float32)

            # Compute absolute difference across all channels
            abs_diff = np.abs(doc_float - template_float)

            # Convert to grayscale by taking mean across BGR channels
            diff_gray = np.mean(abs_diff, axis=2).astype(np.uint8)
            intermediate_images['01_diff_gray'] = diff_gray

            # Calculate mean difference as the metric
            # Higher value = more difference = has content
            mean_diff = np.mean(diff_gray) / 255.0  # Normalize to 0.0-1.0

            # Step 3: Multi-stage content detection to handle pre-printed text residuals
            # Stage 1: Filter out small differences (likely pre-printed text alignment errors)
            significant_threshold = 30  # Only count pixels with diff > 30/255
            significant_diff = diff_gray[diff_gray > significant_threshold]
            significant_ratio = len(significant_diff) / diff_gray.size

            # Stage 2: Calculate metrics
            if len(significant_diff) > 0:
                significant_mean = np.mean(significant_diff) / 255.0
            else:
                significant_mean = 0.0

            # Step 4: Decision logic
            # For fields with pre-printed text (big1, small1), use stricter criteria
            # Heuristic: If mean_diff is very high (>0.15), likely has pre-printed text
            # In this case, require higher significant_ratio to confirm real content
            if mean_diff > 0.15:
                # Likely has pre-printed text residuals
                # Require at least 20% of pixels with significant difference
                has_content = significant_ratio > 0.20
                reasoning = f"mean_diff={mean_diff:.4f} (HIGH, pre-printed?), significant_ratio={significant_ratio:.4f}, threshold=0.20, has_content={has_content}"
            else:
                # Normal fields without pre-printed text
                # Use original simple threshold
                threshold = MIN_ABSOLUTE_DENSITY_THRESHOLD
                has_content = mean_diff > threshold
                reasoning = f"mean_diff={mean_diff:.4f}, threshold={threshold:.4f}, has_content={has_content}"

            # For visualization
            binary_vis = diff_gray
            intermediate_images['02_final'] = binary_vis

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Save debug images and metadata if enabled
            if self.save_debug_images and document_name:
                self._save_intermediate_images(intermediate_images, field_id, document_name)
                result = AIPResult(
                    field_id=field_id,
                    has_content=has_content,
                    ink_ratio=mean_diff,  # Use mean_diff as "ink_ratio"
                    component_count=None,  # Not used
                    processing_time_ms=processing_time_ms
                )
                self._save_metadata(field_id, document_name, result, reasoning)

            return AIPResult(
                field_id=field_id,
                has_content=has_content,
                ink_ratio=mean_diff,  # Use mean_diff as metric
                component_count=None,  # Not applicable
                processing_time_ms=processing_time_ms,
                processed_image=binary_vis  # Difference image for visualization
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"{field_id}: AIP failed: {e}")
            return AIPResult(
                field_id=field_id,
                has_content=None,
                ink_ratio=None,
                component_count=None,
                processing_time_ms=processing_time_ms,
                error_message=str(e)
            )

    def _align_roi_to_template(
        self,
        doc_roi: np.ndarray,
        template_roi: np.ndarray
    ) -> np.ndarray:
        """
        Align document ROI to template ROI using ECC algorithm.

        This performs fine-grained alignment at ROI level to ensure pre-printed
        text perfectly overlaps before template difference. Critical for fields
        with pre-printed text (e.g., big1, small1 stamp fields).

        Args:
            doc_roi: Document ROI image (BGR, uint8)
            template_roi: Blank template ROI image (BGR, uint8)

        Returns:
            Aligned document ROI (BGR, uint8)
        """
        import cv2

        try:
            # Convert to grayscale for alignment
            doc_gray = cv2.cvtColor(doc_roi, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_roi, cv2.COLOR_BGR2GRAY)

            # Define the motion model (MOTION_EUCLIDEAN for translation + rotation)
            warp_mode = cv2.MOTION_EUCLIDEAN

            # Initialize warp matrix (2x3 for Euclidean)
            if warp_mode == cv2.MOTION_EUCLIDEAN:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(3, 3, dtype=np.float32)

            # Set termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

            # Run ECC algorithm
            try:
                (cc, warp_matrix) = cv2.findTransformECC(
                    template_gray,
                    doc_gray,
                    warp_matrix,
                    warp_mode,
                    criteria,
                    None,
                    1  # gaussFiltSize
                )

                # Apply the warp to the original BGR image
                h, w = template_roi.shape[:2]
                aligned_doc = cv2.warpAffine(
                    doc_roi,
                    warp_matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)  # White border
                )

                logger.debug(f"ROI alignment: correlation={cc:.4f}")

                return aligned_doc

            except cv2.error as e:
                # ECC failed (e.g., not enough features), return original
                logger.debug(f"ECC alignment failed: {e}, using original ROI")
                return doc_roi

        except Exception as e:
            logger.warning(f"ROI alignment error: {e}, using original ROI")
            return doc_roi

    def _save_intermediate_images(
        self,
        intermediate_images: Dict[str, np.ndarray],
        field_id: str,
        document_name: str
    ) -> None:
        """
        Save intermediate AIP (Advanced Image Processing) images for debugging.

        Args:
            intermediate_images: Dict mapping step_name -> image
            field_id: Field identifier
            document_name: Document name

        Side Effects:
            - Creates directory: output_dir/{document_name}/{field_id}/
            - Writes PNG files for each intermediate step
        """
        try:
            # Create output directory
            field_dir = self.output_dir / document_name / field_id
            field_dir.mkdir(parents=True, exist_ok=True)

            # Save each intermediate image
            for step_name, image in intermediate_images.items():
                output_path = field_dir / f"{step_name}.png"
                cv2.imwrite(
                    str(output_path),
                    image,
                    [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION_LEVEL]
                )

        except Exception as e:
            logger.error(f"{field_id}: Failed to save intermediate images: {e}")

    def _save_metadata(
        self,
        field_id: str,
        document_name: str,
        result: AIPResult,
        reasoning: str
    ) -> None:
        """
        Save AIP (Advanced Image Processing) metadata JSON for debugging.

        Args:
            field_id: Field identifier
            document_name: Document name
            result: AIPResult with metrics
            reasoning: Decision reasoning string

        Side Effects:
            - Writes {output_dir}/{document_name}/{field_id}/metadata.json
        """
        try:
            field_dir = self.output_dir / document_name / field_id
            metadata_path = field_dir / "metadata.json"

            metadata = {
                "field_id": field_id,
                "has_content": result.has_content,
                "ink_ratio": result.ink_ratio,
                "component_count": result.component_count,
                "processing_time_ms": result.processing_time_ms,
                "thresholds_used": {
                    "MIN_ABSOLUTE_DENSITY_THRESHOLD": MIN_ABSOLUTE_DENSITY_THRESHOLD,
                },
                "decision_reasoning": reasoning,
                "error_message": result.error_message
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"{field_id}: Failed to save metadata: {e}")
