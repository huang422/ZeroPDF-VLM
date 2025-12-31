"""
VLM-based ROI content recognition with exception handling and retry logic.

This module provides the core VLM recognition functionality for extracted ROI regions,
including robust error handling, automatic retry, and validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import json
import time
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from .field_schema import TemplateSchema, FieldSchema, PROMPT_TEMPLATES

# Save intermediate preprocessing images for debugging
DEBUG_SAVE_INTERMEDIATE_IMAGES = os.getenv('DEBUG_ROI_PREPROCESSING', 'false').lower() == 'true'

logger = logging.getLogger(__name__)

# ImageNet normalization constants for InternVL
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    """Build image transformation pipeline with ImageNet normalization.

    Args:
        input_size: Target image size (default: 448)

    Returns:
        Composed torchvision transforms
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list,
                              width: int, height: int, image_size: int = 448):
    """Find closest aspect ratio from target ratios for dynamic preprocessing.

    Args:
        aspect_ratio: Original image aspect ratio (width/height)
        target_ratios: List of (rows, cols) aspect ratio tuples
        width: Original image width
        height: Original image height
        image_size: Base tile size (default: 448)

    Returns:
        Best matching (rows, cols) tuple
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12,
                       image_size: int = 448, use_thumbnail: bool = False):
    """Preprocess image with dynamic aspect ratio splitting.

    Args:
        image: PIL Image to preprocess
        min_num: Minimum number of tiles (default: 1)
        max_num: Maximum number of tiles (default: 12)
        image_size: Size of each tile (default: 448)
        use_thumbnail: Whether to add thumbnail image (default: False)

    Returns:
        List of preprocessed PIL Images (tiles)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target aspect ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split image
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image_from_pil(pil_image: Image.Image, input_size: int = 448, max_num: int = 12):
    """Load and preprocess PIL image for InternVL model.

    Args:
        pil_image: PIL Image to preprocess
        input_size: Base tile size (default: 448)
        max_num: Maximum number of tiles (default: 12)

    Returns:
        torch.Tensor of preprocessed image tiles [num_tiles, 3, H, W]
    """
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


@dataclass
class RecognitionResult:
    """Stores the VLM recognition output for a single field.

    Attributes:
        field_id: Field identifier (links to FieldSchema)
        has_content: True if content detected, False if empty, None for title fields (not applicable)
        content_text: Extracted text/number if has_content==True, null otherwise
        raw_response: Raw VLM output for debugging
        parse_success: True if JSON parsing succeeded, False if fallback to defaults
        inference_time_ms: Time taken for VLM inference in milliseconds
        retry_count: Number of retries attempted (0 = success on first try)
        AIP_has_content: AIP pipeline result (True=filled, False=empty, None=skipped/error)
        processed_roi_image: Final binary image from AIP pipeline for output
    """

    field_id: str
    has_content: Optional[bool]  # None for title fields
    content_text: Optional[str]
    raw_response: str
    parse_success: bool
    inference_time_ms: float
    retry_count: int = 0
    AIP_has_content: Optional[bool] = None  # AIP pipeline result (Feature 004)
    AIP_ink_ratio: Optional[float] = None  # Ink ratio metric (Feature 004) - internal only
    AIP_component_count: Optional[int] = None  # Component count metric (Feature 004) - internal only
    AIP_time_ms: Optional[float] = None  # AIP time (Feature 004) - internal only
    processed_roi_image: Optional[np.ndarray] = None  # AIP ROI image for output (Feature 004)

    def validate(self):
        """Validate recognition result consistency.

        Raises:
            AssertionError: If validation fails
        """
        if self.has_content is False:
            assert self.content_text is None, \
                f"Field {self.field_id}: has_content=False must have content_text=None"

        if self.has_content is None:  # Title field
            assert self.content_text is not None, \
                f"Title field {self.field_id} must have non-null content_text"


@dataclass
class DocumentRecognitionOutput:
    """Aggregates all recognition results for a single document page.

    Attributes:
        document_name: Input filename (e.g., "contractor_form.pdf")
        page_number: Page index (0-based)
        template_id: Matched template (contractor_1, contractor_2, enterprise_1)
        field_results: Recognition results for all fields in template
        results: Overall validation status (True=valid, False=invalid based on validation logic)
        processing_timestamp: Processing timestamp
        total_processing_time_ms: Total time for all fields in this document
    """

    document_name: str
    page_number: int
    template_id: str
    field_results: List[RecognitionResult]
    results: bool  # Validation status calculated from field results
    processing_timestamp: datetime = field(default_factory=datetime.now)
    total_processing_time_ms: float = 0.0

    def validate(self, template_schema: TemplateSchema):
        """Validate document output against template schema.

        Args:
            template_schema: Template schema to validate against

        Raises:
            AssertionError: If validation fails
        """
        assert len(self.field_results) == template_schema.field_count, \
            f"Field count mismatch: expected {template_schema.field_count}, got {len(self.field_results)}"

        for i, (result, schema) in enumerate(zip(self.field_results, template_schema.field_schemas)):
            assert result.field_id == schema.field_id, \
                f"Field ID mismatch at index {i}: expected {schema.field_id}, got {result.field_id}"
            result.validate()

    def calculate_results_status(self) -> bool:
        """Calculate overall results validation status based on field results.

        Logic (updated with AIP priority - Feature 004):
        1. VX1 priority: If VX1.has_content==True (disagreement) using heuristic (NOT AIP), return False
        2. Date fields OR: At least one of (year, month, date) has content (use AIP, fallback to auxiliary, fallback to VLM)
        3. Other fields AND: All non-date, non-title fields have content (use AIP, fallback to auxiliary, fallback to VLM)
        4. Final: date_valid AND other_fields_valid

        Returns:
            True if document passes validation, False otherwise
        """
        # Find VX1 field
        vx1_result = next((r for r in self.field_results if r.field_id == "VX1"), None)

        # VX1 priority check: If VX1 (disagreement checkbox) is checked, document is invalid
        # Use heuristic has_content for VX1, NOT AIP/auxiliary (checkboxes skip both)
        if vx1_result and vx1_result.has_content:
            return False

        # Find date fields (year, month, date) - use AIP_has_content only
        date_fields = [r for r in self.field_results if r.field_id in ("year", "month", "date")]
        date_valid = True
        if date_fields:
            date_valid = any(
                # Use AIP_has_content, fallback to VLM only if AIP is None
                (r.AIP_has_content if r.AIP_has_content is not None else r.has_content)
                for r in date_fields
            )

        # Find other fields (exclude title, VX1, date fields)
        # Note: VX2 is NOT excluded - it must be filled for document to be valid
        # Use AIP_has_content only
        excluded_ids = {"VX1", "year", "month", "date"}
        other_fields = [
            r for r in self.field_results
            if r.has_content is not None  # Exclude title fields
            and r.field_id not in excluded_ids
        ]
        other_fields_valid = all(
            # Use AIP_has_content, fallback to VLM only if AIP is None
            (r.AIP_has_content if r.AIP_has_content is not None else r.has_content)
            for r in other_fields
        )

        result = date_valid and other_fields_valid
        return result

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary for JSON export.

        Returns:
            Dictionary with metadata and nested field results
            Format: {document_ID, results, type, title, processing_timestamp, fields: {...}}
        """
        from .field_schema import TEMPLATE_SCHEMAS

        # Generate document_ID from document_name and page_number
        if self.page_number > 0:
            document_id = f"{self.document_name}_page{self.page_number}"
        else:
            document_id = self.document_name

        # Find title field result
        title_result = next((r for r in self.field_results if r.has_content is None), None)
        title_value = title_result.content_text if title_result else ""

        # Get template schema to check field types
        template_schema = TEMPLATE_SCHEMAS.get(self.template_id)

        # Build nested fields dictionary
        fields = {}
        for result in self.field_results:
            # Skip title field (already in metadata)
            if result.has_content is None:
                continue

            # Find field schema to check type
            field_schema = None
            if template_schema:
                field_schema = template_schema.get_field_by_id(result.field_id)

            # For checkbox/stamp fields, only output has_content (no content_text)
            if field_schema and field_schema.field_type in ["checkbox", "stamp"]:
                fields[result.field_id] = {
                    "VLM_has_content": bool(result.has_content) if result.has_content is not None else result.has_content,
                    "AIP_has_content": bool(result.AIP_has_content) if result.AIP_has_content is not None else result.AIP_has_content
                }
            else:
                # For text/number fields, output both has_content and content_text
                fields[result.field_id] = {
                    "VLM_has_content": bool(result.has_content) if result.has_content is not None else result.has_content,
                    "content_text": result.content_text,
                    "AIP_has_content": bool(result.AIP_has_content) if result.AIP_has_content is not None else result.AIP_has_content
                }

        # Return structured dictionary
        return {
            "document_ID": str(document_id),
            "results": bool(self.results),
            "type": str(self.template_id),
            "title": str(title_value),
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "fields": fields
        }


class VLMRecognizer:
    """VLM-based ROI content recognizer with exception handling and retry.

    This class processes extracted ROI regions using the loaded VLM model and
    generates structured recognition results with automatic retry on failures.

    Checkbox detection is handled by:
    1. AIP pipeline (template difference) -> AIP_has_content
    2. VLM recognition -> has_content
    No heuristic fallback is needed.
    """

    def __init__(self, model: Any, tokenizer: Any, template_schemas: Dict[str, TemplateSchema]):
        """Initialize VLM recognizer.

        Args:
            model: Loaded VLM model (InternVL)
            tokenizer: Loaded VLM tokenizer
            template_schemas: Dictionary mapping template_id to TemplateSchema
        """
        self.model = model
        self.tokenizer = tokenizer
        self.template_schemas = template_schemas

    def _recognize_field(
        self,
        roi_image: np.ndarray,
        field_schema: FieldSchema,
        template_id: str = None,
        blank_template_roi_cache=None,  # BlankTemplateROICache (Feature 004)
        document_name: str = None,  # For debug image saving (Feature 004)
        max_retries: int = 3
    ) -> RecognitionResult:
        """Recognize content in a single ROI field with retry logic.

        Args:
            roi_image: OpenCV BGR image of ROI region
            field_schema: Field schema defining recognition requirements
            template_id: Template identifier for template matching
            blank_template_roi_cache: BlankTemplateROICache for AIP (optional, Feature 004)
            document_name: Document name for debug images (optional, Feature 004)
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            RecognitionResult with recognition output
        """
        # Handle title fields (no VLM inference needed, no auxiliary comparison, no AIP)
        if field_schema.field_type == "title":
            return RecognitionResult(
                field_id=field_schema.field_id,
                has_content=None,  # Not applicable for title
                content_text=field_schema.predefined_value,
                raw_response="",
                parse_success=True,
                inference_time_ms=0.0,
                retry_count=0,
                AIP_has_content=None,  # Skip AIP for title (Feature 004)
                AIP_ink_ratio=None,
                AIP_component_count=None,
                AIP_time_ms=None,
                processed_roi_image=None
            )

        # Get prompt for this field
        prompt = field_schema.get_prompt(PROMPT_TEMPLATES)

        # --- Unified AIP approach for all non-title fields (including checkboxes) ---
        # Step 1: Perform AIP if blank template ROI cache is available (Feature 004)
        aip_result = None
        if blank_template_roi_cache and template_id:
            try:
                from vlm_pdf_recognizer.recognition.roi_preprocessor import ROIPreprocessor

                blank_roi = blank_template_roi_cache.get_blank_roi(template_id, field_schema.field_id)
                if blank_roi is not None:
                    # Create AIP processor with debug image saving if enabled
                    preprocessor = ROIPreprocessor(
                        save_debug_images=DEBUG_SAVE_INTERMEDIATE_IMAGES,
                        output_dir="output/processed_rois" if DEBUG_SAVE_INTERMEDIATE_IMAGES else None
                    )

                    # Run AIP pipeline
                    aip_result = preprocessor.preprocess_roi(
                        roi_image,
                        blank_roi,
                        field_schema.field_id,
                        document_name
                    )

                    density_value = aip_result.ink_ratio if aip_result.ink_ratio is not None else 0.0
                    logger.debug(
                        f"Field {field_schema.field_id}: AIP "
                        f"has_content={aip_result.has_content}, "
                        f"max_density={density_value:.4f}"
                    )
            except Exception as e:
                logger.warning(f"AIP failed for {field_schema.field_id}: {e}")
                aip_result = None

        # Step 2: Run VLM for field recognition
        retry_count = 0
        last_exception = None
        vlm_has_content = None
        vlm_content_text = None
        vlm_raw_response = ""
        vlm_parse_success = False
        vlm_inference_time_ms = 0.0

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                # Pass original image for non-checkbox fields
                vlm_raw_response = self._call_vlm(roi_image, prompt)
                vlm_inference_time_ms = (time.time() - start_time) * 1000

                # Parse JSON response
                vlm_has_content, vlm_content_text, vlm_parse_success = self._parse_vlm_response(vlm_raw_response)
                break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                last_exception = e
                logger.warning(
                    f"Field {field_schema.field_id} VLM recognition failed (attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = 2 ** attempt
                    logger.info(f"Retrying after {delay}s...")
                    time.sleep(delay)

        # Check if all retries failed
        if retry_count >= max_retries:
            logger.error(
                f"Field {field_schema.field_id} VLM recognition failed after {max_retries} attempts: {last_exception}"
            )
            vlm_has_content = False
            vlm_content_text = None
            vlm_raw_response = str(last_exception)
            vlm_parse_success = False

        # Step 3: Return combined result with AIP and VLM outputs
        # For checkbox/stamp fields: content_text must be None when has_content=False
        final_content_text = vlm_content_text
        if field_schema.field_type in ["checkbox", "stamp"] and vlm_has_content is False:
            final_content_text = None

        return RecognitionResult(
            field_id=field_schema.field_id,
            has_content=vlm_has_content,
            content_text=final_content_text,
            raw_response=vlm_raw_response,
            parse_success=vlm_parse_success,
            inference_time_ms=vlm_inference_time_ms,
            retry_count=retry_count,
            AIP_has_content=aip_result.has_content if aip_result else None,
            AIP_ink_ratio=aip_result.ink_ratio if aip_result else None,
            AIP_component_count=aip_result.component_count if aip_result else None,
            AIP_time_ms=aip_result.processing_time_ms if aip_result else None,
            processed_roi_image=aip_result.processed_image if aip_result else None
        )

    def _enhance_checkbox_image(self, roi_image: np.ndarray) -> tuple:
        """Enhance checkbox image to make marks more visible.

        Args:
            roi_image: OpenCV BGR image of checkbox ROI

        Returns:
            Tuple of (binary_image_for_heuristic, enhanced_bgr_for_vlm)
            - binary_image_for_heuristic: Grayscale binary image (0/255) for pixel counting
            - enhanced_bgr_for_vlm: BGR version for VLM input
        """
        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to enhance contrast
        # Use THRESH_BINARY (not INV) so marks are BLACK (0) and background is WHITE (255)
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Use morphological opening to remove small noise artifacts
        # This helps eliminate salt-and-pepper noise after thresholding
        kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel, iterations=1)

        # Convert to BGR for VLM input (keeps the binary effect)
        enhanced_bgr = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

        # Return both: binary grayscale for heuristic, BGR for VLM
        return opened, enhanced_bgr

    def _call_vlm(self, roi_image: np.ndarray, prompt: str) -> str:
        """Call VLM model for inference.

        Args:
            roi_image: OpenCV BGR image of ROI region
            prompt: Traditional Chinese prompt text

        Returns:
            Raw VLM response text

        Raises:
            RuntimeError: If image preprocessing or VLM inference fails
        """
        try:
            # Step 1: Convert OpenCV BGR to PIL RGB
            pil_image = self._opencv_to_pil(roi_image)

            # Step 2: Preprocess image for InternVL with dynamic resolution
            pixel_values = load_image_from_pil(pil_image, input_size=448, max_num=12)
            pixel_values = pixel_values.to(self.model.dtype).to(self.model.device)

            # Step 3: Run VLM inference with deterministic generation config
            generation_config = {
                "max_new_tokens": 256,
                "do_sample": False,  # Deterministic output for JSON parsing
                "temperature": 0.0   # No randomness
            }

            response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)

            return response

        except Exception as e:
            raise RuntimeError(f"VLM inference failed: {e}")

    def _opencv_to_pil(self, opencv_image: np.ndarray):
        """Convert OpenCV BGR image to PIL RGB image.

        Args:
            opencv_image: OpenCV BGR format numpy array

        Returns:
            PIL Image in RGB format

        Raises:
            RuntimeError: If conversion fails
        """
        try:
            import cv2
            from PIL import Image

            # Convert BGR to RGB
            rgb_array = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(rgb_array)

            return pil_image

        except Exception as e:
            raise RuntimeError(f"Failed to convert OpenCV image to PIL: {e}")

    def _parse_vlm_response(self, raw_response: str) -> tuple[bool, Optional[str], bool]:
        """Parse VLM JSON response to extract has_content and content_text.

        Args:
            raw_response: Raw VLM output text

        Returns:
            Tuple of (has_content, content_text, parse_success)
        """
        try:
            # Try direct JSON parsing first
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError as first_error:
                # Extract JSON from response (may have surrounding text)
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1

                if json_start == -1 or json_end == 0:
                    logger.warning(f"No JSON object found in response: {raw_response}")
                    return False, None, False

                json_str = raw_response[json_start:json_end]

                # Try parsing extracted JSON
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as second_error:
                    # Fix invalid escape sequences in JSON string
                    # VLM sometimes outputs invalid escapes like \某 instead of properly escaping backslashes
                    logger.debug(f"JSON parse failed, attempting to fix escape sequences: {second_error}")

                    # Fix common invalid escape patterns
                    # Replace single backslashes with double backslashes, except for valid JSON escapes
                    import re
                    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
                    # Replace any backslash not followed by these valid escapes
                    fixed_json = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

                    try:
                        data = json.loads(fixed_json)
                        logger.debug(f"Successfully parsed after escape sequence fix")
                    except json.JSONDecodeError as third_error:
                        logger.warning(f"Failed to parse even after escape fix: {third_error}")
                        logger.debug(f"Original JSON: {json_str}")
                        logger.debug(f"Fixed JSON: {fixed_json}")
                        return False, None, False

            # Extract fields
            has_content = data.get("has_content", False)
            content_text = data.get("content_text", None)

            # Handle "null" string as None
            if content_text == "null" or content_text == "None":
                content_text = None

            return has_content, content_text, True

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse VLM response as JSON: {e}")
            logger.debug(f"Raw response: {raw_response}")

            # Fallback to defaults
            return False, None, False

    def process_document(
        self,
        roi_images: List[np.ndarray],
        template_id: str,
        page_number: int,
        document_name: str,
        blank_template_roi_cache=None  # BlankTemplateROICache (Feature 004)
    ) -> DocumentRecognitionOutput:
        """Process all ROI fields for a single document page.

        Args:
            roi_images: List of OpenCV BGR ROI images in template order
            template_id: Template identifier (contractor_1, contractor_2, enterprise_1)
            page_number: Page index (0-based)
            document_name: Input filename
            blank_template_roi_cache: BlankTemplateROICache for AIP (optional, Feature 004)

        Returns:
            DocumentRecognitionOutput with all field results and validation status

        Raises:
            ValueError: If template_id not found or ROI count mismatch
        """
        if template_id not in self.template_schemas:
            raise ValueError(f"Template ID '{template_id}' not found in schemas")

        template_schema = self.template_schemas[template_id]

        if len(roi_images) != template_schema.field_count:
            raise ValueError(
                f"ROI count mismatch: expected {template_schema.field_count}, got {len(roi_images)}"
            )

        logger.info(f"Processing document '{document_name}' page {page_number} with template '{template_id}'")

        start_time = time.time()
        field_results = []

        # Process each ROI field
        for roi_image, field_schema in zip(roi_images, template_schema.field_schemas):
            result = self._recognize_field(
                roi_image,
                field_schema,
                template_id,
                blank_template_roi_cache,
                document_name
            )
            field_results.append(result)
            logger.debug(
                f"Field {field_schema.field_id}: has_content={result.has_content}, "
                f"content_text={result.content_text}, inference_time={result.inference_time_ms:.2f}ms"
            )

        total_time_ms = (time.time() - start_time) * 1000

        # Create document output
        doc_output = DocumentRecognitionOutput(
            document_name=document_name,
            page_number=page_number,
            template_id=template_id,
            field_results=field_results,
            results=False,  # Will be calculated
            processing_timestamp=datetime.now(),
            total_processing_time_ms=total_time_ms
        )

        # Calculate validation status
        doc_output.results = doc_output.calculate_results_status()

        # Validate against template schema
        doc_output.validate(template_schema)

        logger.info(
            f"Document '{document_name}' page {page_number} processed: "
            f"results={doc_output.results}, time={total_time_ms:.2f}ms"
        )

        return doc_output
