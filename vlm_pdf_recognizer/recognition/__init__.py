"""
VLM-based ROI content recognition module.

This module provides Vision Language Model (VLM) integration for recognizing
content within extracted ROI regions from PDF documents.

Components:
- vlm_loader: Hardware-adaptive VLM model loading (InternVL 3.5-1B)
- field_schema: Template-specific field definitions and prompt templates
- vlm_recognizer: Core recognition logic with exception handling and retry

Note: JSON export functionality has been integrated into vlm_pdf_recognizer.output module.
"""

from .vlm_loader import VLMLoader, VLMConfig
from .field_schema import FieldSchema, TemplateSchema, TEMPLATE_SCHEMAS, PROMPT_TEMPLATES
from .vlm_recognizer import VLMRecognizer, RecognitionResult, DocumentRecognitionOutput
from .csv_exporter import export_recognition_results_to_csv

__all__ = [
    'VLMLoader',
    'VLMConfig',
    'FieldSchema',
    'TemplateSchema',
    'VLMRecognizer',
    'RecognitionResult',
    'DocumentRecognitionOutput',
    'TEMPLATE_SCHEMAS',
    'PROMPT_TEMPLATES',
    'export_recognition_results_to_csv',
]
