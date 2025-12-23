"""Alignment module for SIFT feature extraction, matching, and geometric correction."""

from .feature_extractor import extract_sift_features, extract_template_features, extract_document_features
from .template_matcher import match_templates, UnknownDocumentError, TemplateMatchResult, FeatureMatch
from .geometric_corrector import align_document_to_template, AlignmentError

__all__ = [
    'extract_sift_features',
    'extract_template_features',
    'extract_document_features',
    'match_templates',
    'UnknownDocumentError',
    'TemplateMatchResult',
    'FeatureMatch',
    'align_document_to_template',
    'AlignmentError'
]
