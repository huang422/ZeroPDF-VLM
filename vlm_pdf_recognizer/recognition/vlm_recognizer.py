"""
VLM-based ROI content recognition using Ollama API with exception handling and retry logic.

This module provides the core VLM recognition functionality for extracted ROI regions,
including robust error handling, automatic retry, and validation.
Uses Ollama glm-ocr model for inference via HTTP API.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import re
import time
import os
import base64

import cv2
import numpy as np
from opencc import OpenCC

from .field_schema import TemplateSchema, FieldSchema, PROMPT_TEMPLATES

# Simplified Chinese to Traditional Chinese converter (singleton)
_s2t_converter = OpenCC('s2t')

# Save intermediate preprocessing images for debugging
DEBUG_SAVE_INTERMEDIATE_IMAGES = os.getenv('DEBUG_ROI_PREPROCESSING', 'false').lower() == 'true'

logger = logging.getLogger(__name__)


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
    VLM_content_text_allPdf: Optional[str] = None  # Full-page VLM recognition result

    def validate(self):
        """Validate recognition result consistency.

        Raises:
            AssertionError: If validation fails
        """
        if self.has_content is False:
            assert self.content_text is None, \
                f"Field {self.field_id}: has_content=False must have content_text=None"


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
        case_id: Case identifier from directory structure (optional)
    """

    document_name: str
    page_number: int
    template_id: str
    field_results: List[RecognitionResult]
    results: bool  # Validation status calculated from field results
    processing_timestamp: datetime = field(default_factory=datetime.now)
    total_processing_time_ms: float = 0.0
    case_id: Optional[str] = None  # Case identifier from input directory structure

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

        has_content is determined by AIP (not VLM). VLM is only used for text content.

        Logic:
        1. VX1 priority: If VX1.has_content==True (disagreement checkbox checked), return False
        2. Date fields OR: At least one of (year, month, date) has content
        3. Other fields AND: All non-date, non-title, non-version fields have content
        4. Final: date_valid AND other_fields_valid

        Returns:
            True if document passes validation, False otherwise
        """
        # VX1 priority check: If VX1 (disagreement checkbox) is checked, document is invalid
        vx1_result = next((r for r in self.field_results if r.field_id == "VX1"), None)
        if vx1_result and vx1_result.has_content:
            return False

        # Date fields OR: at least one of (year, month, date) has content
        date_fields = [r for r in self.field_results if r.field_id in ("year", "month", "date")]
        date_valid = True
        if date_fields:
            date_valid = any(r.has_content for r in date_fields)

        # Other fields AND: all must have content
        # Exclude: title (has_content=None), checkboxes (special logic via VX1),
        #          date fields (OR logic above), version (always True)
        from .field_schema import TEMPLATE_SCHEMAS
        template_schema = TEMPLATE_SCHEMAS.get(self.template_id)

        excluded_ids = {"VX1", "VX2", "year", "month", "date"}
        other_fields = [
            r for r in self.field_results
            if r.has_content is not None  # Exclude title fields
            and r.field_id not in excluded_ids
        ]

        # Exclude version and checkbox fields from validation
        if template_schema:
            other_fields = [
                r for r in other_fields
                if not (template_schema.get_field_by_id(r.field_id) and
                        template_schema.get_field_by_id(r.field_id).field_type in ("version", "checkbox"))
            ]

        other_fields_valid = all(r.has_content for r in other_fields)

        return date_valid and other_fields_valid

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary for JSON export.

        Returns:
            Dictionary with metadata and nested field results
            Format: {document_ID, results, type, title, version, case_id, processing_timestamp, fields: {...}}
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

        # Find version field result (extract to top level, next to title)
        version_value = None
        for result in self.field_results:
            if template_schema:
                field_schema = template_schema.get_field_by_id(result.field_id)
                if field_schema and field_schema.field_type == "version":
                    version_value = result.content_text if result.content_text else ""
                    break

        # Build nested fields dictionary
        fields = {}
        for result in self.field_results:
            # Find field schema to check type
            field_schema = None
            if template_schema:
                field_schema = template_schema.get_field_by_id(result.field_id)

            # Skip title field (already in metadata)
            if field_schema and field_schema.field_type == "title":
                continue

            # Skip version field (moved to top level, next to title)
            if field_schema and field_schema.field_type == "version":
                continue

            # Build field data - has_content from AIP (no separate VLM_has_content/AIP_has_content)
            field_data = {
                "has_content": bool(result.has_content) if result.has_content is not None else result.has_content,
            }

            # For all non-checkbox fields, output content_text (stamp, text, number, person_number)
            if field_schema and field_schema.field_type not in ["checkbox"]:
                field_data["content_text"] = result.content_text

            # Add VLM_content_text_allPdf if available
            if result.VLM_content_text_allPdf is not None:
                field_data["VLM_content_text_allPdf"] = result.VLM_content_text_allPdf

            fields[result.field_id] = field_data

        # Return structured dictionary with version at top level (after title)
        result_dict = {
            "document_ID": str(document_id),
            "results": bool(self.results),
            "type": str(self.template_id),
            "title": str(title_value),
            "version": str(version_value) if version_value is not None else "",
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "fields": fields
        }

        # Add case_id if available
        if self.case_id:
            result_dict["case_id"] = self.case_id

        return result_dict


class VLMRecognizer:
    """VLM-based ROI content recognizer using Ollama API with exception handling and retry.

    This class processes extracted ROI regions using the Ollama glm-ocr model and
    generates structured recognition results with automatic retry on failures.
    """

    def __init__(self, client: Any, tokenizer: Any, template_schemas: Dict[str, TemplateSchema]):
        """Initialize VLM recognizer.

        Args:
            client: OllamaClient instance
            tokenizer: Not used (kept for API compatibility), pass None
            template_schemas: Dictionary mapping template_id to TemplateSchema
        """
        self.client = client
        self.template_schemas = template_schemas

        # Get config from VLMLoader singleton for model settings
        from .vlm_loader import VLMLoader
        loader = VLMLoader.get_instance()
        self._config = loader.config

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
        # Handle title fields (no AIP, no VLM)
        if field_schema.field_type == "title":
            return RecognitionResult(
                field_id=field_schema.field_id,
                has_content=None,  # Not applicable for title
                content_text=field_schema.predefined_value,
                raw_response="",
                parse_success=True,
                inference_time_ms=0.0,
                retry_count=0,
                AIP_has_content=None,
                AIP_ink_ratio=None,
                AIP_component_count=None,
                AIP_time_ms=None,
                processed_roi_image=None
            )

        # --- Step 1: AIP for has_content (all non-title, non-version fields) ---
        # AIP determines whether the field has content or not.
        # Version fields skip AIP (always have content).
        aip_result = None
        if field_schema.field_type != "version" and blank_template_roi_cache and template_id:
            try:
                from vlm_pdf_recognizer.recognition.roi_preprocessor import ROIPreprocessor

                blank_roi = blank_template_roi_cache.get_blank_roi(template_id, field_schema.field_id)
                if blank_roi is not None:
                    preprocessor = ROIPreprocessor(
                        save_debug_images=DEBUG_SAVE_INTERMEDIATE_IMAGES,
                        output_dir="output/processed_rois" if DEBUG_SAVE_INTERMEDIATE_IMAGES else None
                    )

                    aip_result = preprocessor.preprocess_roi(
                        roi_image, blank_roi, field_schema.field_id, document_name
                    )

                    density_value = aip_result.ink_ratio if aip_result.ink_ratio is not None else 0.0
                    logger.debug(
                        f"Field {field_schema.field_id}: AIP "
                        f"has_content={aip_result.has_content}, "
                        f"ink_ratio={density_value:.4f}"
                    )
            except Exception as e:
                logger.warning(f"AIP failed for {field_schema.field_id}: {e}")
                aip_result = None

        # Determine has_content from AIP
        if field_schema.field_type == "version":
            has_content = True  # Version field always has content
        elif aip_result is not None:
            has_content = aip_result.has_content
        else:
            has_content = None  # AIP not available

        # --- Step 2: VLM for content_text ---
        # Skip VLM for checkbox (only need has_content from AIP)
        # Stamp fields: run VLM to recognize stamp text content (e.g., company name)
        # Skip VLM if AIP says no content (content_text=None)
        vlm_content_text = None
        vlm_raw_response = ""
        vlm_parse_success = True
        vlm_inference_time_ms = 0.0
        retry_count = 0

        needs_vlm = field_schema.field_type not in ["checkbox"]
        if needs_vlm and has_content is False:
            needs_vlm = False

        if needs_vlm:
            prompt = field_schema.get_prompt(PROMPT_TEMPLATES)
            last_exception = None

            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    vlm_raw_response = self._call_vlm(roi_image, prompt)
                    vlm_inference_time_ms = (time.time() - start_time) * 1000

                    # Parse raw text response from glm-ocr
                    vlm_content_text, vlm_parse_success = self._parse_vlm_response(vlm_raw_response)
                    break  # Success

                except Exception as e:
                    retry_count += 1
                    last_exception = e
                    logger.warning(
                        f"Field {field_schema.field_id} VLM failed "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)

            if retry_count >= max_retries:
                logger.error(
                    f"Field {field_schema.field_id} VLM failed after "
                    f"{max_retries} attempts: {last_exception}"
                )
                vlm_content_text = None
                vlm_raw_response = str(last_exception)
                vlm_parse_success = False

        # Step 3: Clean content text (remove $$, format numbers, convert to 繁體中文)
        if vlm_content_text is not None:
            vlm_content_text = self._clean_content_text(vlm_content_text, field_schema.field_type)

        return RecognitionResult(
            field_id=field_schema.field_id,
            has_content=has_content,
            content_text=vlm_content_text,
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

    def _call_vlm(self, roi_image: np.ndarray, prompt: str) -> str:
        """Call Ollama glm-ocr model for inference.

        Args:
            roi_image: OpenCV BGR image of ROI region
            prompt: Traditional Chinese prompt text

        Returns:
            Raw VLM response text

        Raises:
            RuntimeError: If image encoding or Ollama API call fails
        """
        try:
            # Step 1: Encode image to base64 PNG
            image_b64 = self._encode_image_base64(roi_image)

            # Step 2: Call Ollama generate API with image
            model_name = self._config.model_name if self._config else DEFAULT_MODEL_NAME
            temperature = self._config.temperature if self._config else 0.0
            num_predict = self._config.num_predict if self._config else 256
            num_gpu = self._config.num_gpu if self._config else -1

            from .vlm_loader import DEFAULT_MODEL_NAME

            response = self.client.generate(
                model=model_name,
                prompt=prompt,
                images=[image_b64],
                temperature=temperature,
                num_predict=num_predict,
                num_gpu=num_gpu
            )

            return response

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"VLM inference failed: {e}")

    def _encode_image_base64(self, opencv_image: np.ndarray) -> str:
        """Encode OpenCV BGR image to base64 PNG string for Ollama API.

        Args:
            opencv_image: OpenCV BGR format numpy array

        Returns:
            Base64-encoded PNG image string

        Raises:
            RuntimeError: If encoding fails
        """
        try:
            # Encode as PNG
            success, buffer = cv2.imencode('.png', opencv_image)
            if not success:
                raise RuntimeError("Failed to encode image to PNG")

            # Convert to base64
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            return image_b64

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Failed to encode image to base64: {e}")

    def _parse_vlm_response(self, raw_response: str) -> tuple:
        """Parse raw text response from glm-ocr.

        glm-ocr is a pure OCR model that returns raw text (not JSON).
        Empty/whitespace response means no content was detected in the image.

        Args:
            raw_response: Raw OCR text output from glm-ocr

        Returns:
            Tuple of (content_text, parse_success)
            - content_text: Stripped text string, or None if empty
            - parse_success: Always True for raw text parsing
        """
        if not raw_response or not raw_response.strip():
            return None, True

        content_text = raw_response.strip()
        return content_text, True

    def _clean_content_text(self, content_text: str, field_type: str) -> str:
        """Clean and normalize VLM output text based on field type.

        Post-processing steps:
        1. Strip prompt echo: detect and remove prompt text leaked into output
        2. Remove LaTeX $$ markers from OCR output
        3. Remove markdown formatting (```markdown ... ```)
        4. Convert Simplified Chinese to Traditional Chinese (繁體中文)
        5. Field-type-specific formatting (numbers, person_number, etc.)

        Args:
            content_text: Raw text from VLM (already stripped)
            field_type: Field type for type-specific cleaning

        Returns:
            Cleaned text string, or None if empty after cleaning
        """
        if not content_text:
            return content_text

        text = content_text

        # Step 1: Strip prompt echo - detect prompt fragments in output
        # glm-ocr sometimes echoes the prompt text back as part of the response
        _prompt_fragments = [
            "請辨識", "請描述", "辨識對象", "觀察重點", "格式要求",
            "辨識步驟", "忽略以下", "嚴格排除", "常見內容", "常見格式",
            "只輸出數字", "只輸出字母", "注意：", "辨識要求",
            "手寫文字筆跡", "印刷文字", "手寫數字筆跡", "印刷數字",
            "繁體中文文字內容", "繁體中文文字",
        ]
        for fragment in _prompt_fragments:
            if fragment in text:
                logger.warning(
                    f"Prompt echo detected in VLM output (fragment: '{fragment}'). "
                    f"Discarding output: {text[:80]}..."
                )
                return None

        # Step 2: Remove markdown formatting (```markdown ... ```)
        text = re.sub(r'```\w*\s*', '', text)
        text = re.sub(r'```', '', text)
        text = text.strip()

        # Step 3: Remove LaTeX $$ markers (glm-ocr sometimes wraps output in $$...$$)
        text = re.sub(r'^\$\$\s*', '', text)
        text = re.sub(r'\s*\$\$$', '', text)
        text = text.strip()

        # Step 4: Convert Simplified Chinese to Traditional Chinese (MUST be before placeholder removal)
        text = _s2t_converter.convert(text)

        # Step 5: Remove predefined stamp placeholder text (after S2T conversion)
        # These are pre-printed labels on the document template, NOT actual stamp content
        _stamp_placeholders = [
            "負責人蓋章處", "公司大章蓋章處", "公司小章蓋章處",
            "蓋章處", "簽章處", "用印處",
        ]
        for placeholder in _stamp_placeholders:
            text = text.replace(placeholder, "")
        text = text.strip()

        # Step 6: Field-type-specific cleaning
        if field_type in ("version", "number"):
            # Version/number fields: keep only digits
            text = re.sub(r'[^\d]', '', text)
        elif field_type == "person_number":
            # Person number (e.g., ID number): keep letters and digits only
            text = re.sub(r'[^A-Za-z0-9]', '', text)
        elif field_type == "stamp":
            # Stamp fields: normalize whitespace, keep Traditional Chinese text
            text = re.sub(r'\s+', '', text)
        else:
            # Text fields: normalize whitespace (collapse multiple spaces/newlines)
            text = re.sub(r'\s+', '', text)

        return text if text else None

    def process_document(
        self,
        roi_images: List[np.ndarray],
        template_id: str,
        page_number: int,
        document_name: str,
        blank_template_roi_cache=None,  # BlankTemplateROICache (Feature 004)
        case_id: str = None  # Case identifier from directory structure
    ) -> DocumentRecognitionOutput:
        """Process all ROI fields for a single document page.

        Args:
            roi_images: List of OpenCV BGR ROI images in template order
            template_id: Template identifier (contractor_1, contractor_2, enterprise_1)
            page_number: Page index (0-based)
            document_name: Input filename
            blank_template_roi_cache: BlankTemplateROICache for AIP (optional, Feature 004)
            case_id: Case identifier from input directory structure (optional)

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
            total_processing_time_ms=total_time_ms,
            case_id=case_id
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
