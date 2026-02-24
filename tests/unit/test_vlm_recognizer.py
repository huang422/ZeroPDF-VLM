"""Unit tests for VLM recognizer with mocked VLM responses."""

import pytest
import numpy as np
from datetime import datetime
from vlm_pdf_recognizer.recognition.vlm_recognizer import (
    RecognitionResult,
    DocumentRecognitionOutput
)
from vlm_pdf_recognizer.recognition.field_schema import TEMPLATE_SCHEMAS


class TestRecognitionResult:
    """Test RecognitionResult dataclass and validation."""

    def test_valid_non_title_result(self):
        """Non-title result with has_content=True (from AIP) should validate."""
        result = RecognitionResult(
            field_id="test_field",
            has_content=True,
            content_text="Test content",
            raw_response="Test content",
            parse_success=True,
            inference_time_ms=100.0,
            retry_count=0
        )
        result.validate()  # Should not raise

    def test_empty_field_result(self):
        """Field with has_content=False (from AIP) must have content_text=None."""
        result = RecognitionResult(
            field_id="test_field",
            has_content=False,
            content_text=None,
            raw_response="",
            parse_success=True,
            inference_time_ms=0.0,
            retry_count=0
        )
        result.validate()  # Should not raise

    def test_empty_field_with_text_fails(self):
        """Field with has_content=False and non-null text should fail."""
        result = RecognitionResult(
            field_id="test_field",
            has_content=False,
            content_text="Invalid",
            raw_response="",
            parse_success=True,
            inference_time_ms=0.0,
            retry_count=0
        )
        with pytest.raises(AssertionError, match="must have content_text=None"):
            result.validate()

    def test_title_field_result(self):
        """Title field with has_content=None should validate."""
        result = RecognitionResult(
            field_id="contractor_1_title",
            has_content=None,  # Title field
            content_text="企業負責人電信信評報告之使用授權書",
            raw_response="",
            parse_success=True,
            inference_time_ms=0.0,
            retry_count=0
        )
        result.validate()  # Should not raise

    def test_aip_unavailable_result(self):
        """Field with has_content=None (AIP unavailable) should validate."""
        result = RecognitionResult(
            field_id="test_field",
            has_content=None,
            content_text=None,
            raw_response="",
            parse_success=True,
            inference_time_ms=0.0,
            retry_count=0
        )
        result.validate()  # Should not raise


class TestDocumentRecognitionOutput:
    """Test DocumentRecognitionOutput dataclass and validation."""

    def test_valid_document_output(self):
        """Document output with correct field count should validate."""
        template_schema = TEMPLATE_SCHEMAS["contractor_2"]  # 3 fields (title, small1, version)

        results = [
            RecognitionResult(
                field_id="contractor_2_title",
                has_content=None,
                content_text="個人資料特定目的外用告知事項暨同意書",
                raw_response="",
                parse_success=True,
                inference_time_ms=0.0,
                retry_count=0
            ),
            RecognitionResult(
                field_id="small1",
                has_content=True,
                content_text=None,
                raw_response="",
                parse_success=True,
                inference_time_ms=0.0,
                retry_count=0
            ),
            RecognitionResult(
                field_id="version",
                has_content=True,
                content_text="20250417",
                raw_response="20250417",
                parse_success=True,
                inference_time_ms=100.0,
                retry_count=0
            ),
        ]

        doc_output = DocumentRecognitionOutput(
            document_name="test.pdf",
            page_number=0,
            template_id="contractor_2",
            field_results=results,
            results=True,
            processing_timestamp=datetime.now(),
            total_processing_time_ms=150.0
        )

        doc_output.validate(template_schema)  # Should not raise

    def test_field_count_mismatch_fails(self):
        """Document with wrong field count should fail validation."""
        template_schema = TEMPLATE_SCHEMAS["contractor_2"]  # Expects 3 fields

        results = [
            RecognitionResult(
                field_id="contractor_2_title",
                has_content=None,
                content_text="Test",
                raw_response="",
                parse_success=True,
                inference_time_ms=0.0,
                retry_count=0
            )
            # Missing other fields
        ]

        doc_output = DocumentRecognitionOutput(
            document_name="test.pdf",
            page_number=0,
            template_id="contractor_2",
            field_results=results,
            results=True,
            processing_timestamp=datetime.now(),
            total_processing_time_ms=0.0
        )

        with pytest.raises(AssertionError, match="Field count mismatch"):
            doc_output.validate(template_schema)

    def test_calculate_results_vx1_true_returns_false(self):
        """If VX1.has_content=True (disagreement), results should be False."""
        template_schema = TEMPLATE_SCHEMAS["contractor_1"]

        # Create minimal results with VX1=True
        results = []
        for field in template_schema.field_schemas:
            if field.field_type == "title":
                result = RecognitionResult(
                    field_id=field.field_id,
                    has_content=None,
                    content_text=field.predefined_value,
                    raw_response="",
                    parse_success=True,
                    inference_time_ms=0.0,
                    retry_count=0
                )
            elif field.field_id == "VX1":
                result = RecognitionResult(
                    field_id=field.field_id,
                    has_content=True,  # Disagreement
                    content_text=None,
                    raw_response='{"has_content": true, "content_text": null}',
                    parse_success=True,
                    inference_time_ms=100.0,
                    retry_count=0
                )
            else:
                result = RecognitionResult(
                    field_id=field.field_id,
                    has_content=True,
                    content_text="Test",
                    raw_response='{"has_content": true, "content_text": "Test"}',
                    parse_success=True,
                    inference_time_ms=100.0,
                    retry_count=0
                )
            results.append(result)

        doc_output = DocumentRecognitionOutput(
            document_name="test.pdf",
            page_number=0,
            template_id="contractor_1",
            field_results=results,
            results=False,  # Will be recalculated
            processing_timestamp=datetime.now(),
            total_processing_time_ms=1000.0
        )

        calculated_results = doc_output.calculate_results_status()
        assert calculated_results is False, "VX1=True should set results=False"

    def test_calculate_results_all_valid_returns_true(self):
        """If VX1=False, dates valid, other fields valid, results should be True."""
        template_schema = TEMPLATE_SCHEMAS["contractor_1"]

        results = []
        for field in template_schema.field_schemas:
            if field.field_type == "title":
                result = RecognitionResult(
                    field_id=field.field_id,
                    has_content=None,
                    content_text=field.predefined_value,
                    raw_response="",
                    parse_success=True,
                    inference_time_ms=0.0,
                    retry_count=0
                )
            elif field.field_id in ["VX1", "VX2"]:
                result = RecognitionResult(
                    field_id=field.field_id,
                    has_content=False,  # Agreement
                    content_text=None,
                    raw_response='{"has_content": false, "content_text": null}',
                    parse_success=True,
                    inference_time_ms=100.0,
                    retry_count=0
                )
            else:
                result = RecognitionResult(
                    field_id=field.field_id,
                    has_content=True,
                    content_text="Test",
                    raw_response='{"has_content": true, "content_text": "Test"}',
                    parse_success=True,
                    inference_time_ms=100.0,
                    retry_count=0
                )
            results.append(result)

        doc_output = DocumentRecognitionOutput(
            document_name="test.pdf",
            page_number=0,
            template_id="contractor_1",
            field_results=results,
            results=False,
            processing_timestamp=datetime.now(),
            total_processing_time_ms=1300.0
        )

        calculated_results = doc_output.calculate_results_status()
        assert calculated_results is True, "All valid fields should set results=True"

    def test_to_json_dict(self):
        """to_json_dict should produce correct nested dictionary."""
        results = [
            RecognitionResult(
                field_id="contractor_2_title",
                has_content=None,
                content_text="個人資料特定目的外用告知事項暨同意書",
                raw_response="",
                parse_success=True,
                inference_time_ms=0.0,
                retry_count=0
            ),
            RecognitionResult(
                field_id="small1",
                has_content=True,
                content_text=None,
                raw_response="",
                parse_success=True,
                inference_time_ms=0.0,
                retry_count=0
            ),
            RecognitionResult(
                field_id="version",
                has_content=True,
                content_text="20250417",
                raw_response="20250417",
                parse_success=True,
                inference_time_ms=100.0,
                retry_count=0
            ),
        ]

        doc_output = DocumentRecognitionOutput(
            document_name="test.pdf",
            page_number=0,
            template_id="contractor_2",
            field_results=results,
            results=True,
            processing_timestamp=datetime(2025, 12, 29, 10, 30, 45),
            total_processing_time_ms=150.0
        )

        json_dict = doc_output.to_json_dict()

        # Check metadata
        assert json_dict["document_ID"] == "test.pdf"
        assert json_dict["type"] == "contractor_2"
        assert json_dict["title"] == "個人資料特定目的外用告知事項暨同意書"
        assert json_dict["version"] == "20250417"
        assert json_dict["results"] is True
        assert "processing_timestamp" in json_dict

        # Check fields structure
        assert "fields" in json_dict
        assert "small1" in json_dict["fields"]

        # stamp field should have has_content AND content_text, no VLM_has_content/AIP_has_content
        assert json_dict["fields"]["small1"]["has_content"] is True
        assert "content_text" in json_dict["fields"]["small1"]
        assert "VLM_has_content" not in json_dict["fields"]["small1"]
        assert "AIP_has_content" not in json_dict["fields"]["small1"]

        # version field should be at top level, not in fields
        assert "version" not in json_dict["fields"]
