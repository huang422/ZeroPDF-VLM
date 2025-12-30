"""Unit tests for output module including VLM JSON export."""

import pytest
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime

from vlm_pdf_recognizer.output import save_vlm_recognition_results, append_vlm_recognition_result, validate_vlm_json_structure
from vlm_pdf_recognizer.recognition.vlm_recognizer import RecognitionResult, DocumentRecognitionOutput


class TestVLMJSONExport:
    """Test VLM JSON export functionality integrated in output module."""

    def create_sample_document_output(self, template_id="contractor_2"):
        """Create sample DocumentRecognitionOutput for testing."""
        if template_id == "contractor_2":
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
                    field_id="small",
                    has_content=True,
                    content_text=None,
                    raw_response='{"has_content": true, "content_text": null}',
                    parse_success=True,
                    inference_time_ms=150.0,
                    retry_count=0
                )
            ]
        else:
            raise ValueError(f"Template {template_id} not implemented in test")

        return DocumentRecognitionOutput(
            document_name="test.pdf",
            page_number=0,
            template_id=template_id,
            field_results=results,
            results=True,
            processing_timestamp=datetime(2025, 12, 29, 10, 30, 45),
            total_processing_time_ms=150.0
        )

    def test_save_single_document(self):
        """Save single document to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_output = self.create_sample_document_output()

            json_path = save_vlm_recognition_results([doc_output], tmpdir)

            # Verify file exists
            assert os.path.exists(json_path)

            # Verify content
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data) == 1
            record = data[0]
            assert record["document_ID"] == "test.pdf"
            assert record["type"] == "contractor_2"
            assert record["title"] == "個人資料特定目的外用告知事項暨同意書"
            assert record["results"] is True
            assert "fields" in record
            assert "small" in record["fields"]

    def test_save_multiple_documents(self):
        """Save multiple documents to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 document outputs
            doc_outputs = []
            for i in range(3):
                doc_output = self.create_sample_document_output()
                doc_output.document_name = f"test_{i}.pdf"
                doc_output.page_number = i
                doc_outputs.append(doc_output)

            json_path = save_vlm_recognition_results(doc_outputs, tmpdir)

            # Verify content
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data) == 3
            assert data[0]["document_ID"] == "test_0.pdf"
            assert data[1]["document_ID"] == "test_1.pdf_page1"
            assert data[2]["document_ID"] == "test_2.pdf_page2"

    def test_save_empty_list_raises(self):
        """Saving empty list should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="empty"):
                save_vlm_recognition_results([], tmpdir)

    def test_json_encoding_utf8(self):
        """JSON should be exported with UTF-8 encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_output = self.create_sample_document_output()

            json_path = save_vlm_recognition_results([doc_output], tmpdir)

            # Verify UTF-8 encoding by loading with Chinese characters
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert data[0]["title"] == "個人資料特定目的外用告知事項暨同意書"

    def test_append_to_new_file(self):
        """Appending to new file should create it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_output = self.create_sample_document_output()

            json_path = append_vlm_recognition_result(doc_output, tmpdir)

            # Verify file exists
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["document_ID"] == "test.pdf"

    def test_append_to_existing_file(self):
        """Appending to existing file should add to array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial file
            doc_output1 = self.create_sample_document_output()
            doc_output1.document_name = "doc1.pdf"
            append_vlm_recognition_result(doc_output1, tmpdir)

            # Append second document
            doc_output2 = self.create_sample_document_output()
            doc_output2.document_name = "doc2.pdf"
            json_path = append_vlm_recognition_result(doc_output2, tmpdir)

            # Verify content
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["document_ID"] == "doc1.pdf"
            assert data[1]["document_ID"] == "doc2.pdf"

    def test_validate_json_valid(self):
        """validate_vlm_json_structure should return True for valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_output = self.create_sample_document_output("contractor_2")

            json_path = save_vlm_recognition_results([doc_output], tmpdir)

            # Validate
            is_valid = validate_vlm_json_structure(json_path, "contractor_2")
            assert is_valid is True

    def test_validate_json_invalid_template(self):
        """validate_vlm_json_structure should return False for unknown template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_output = self.create_sample_document_output()

            json_path = save_vlm_recognition_results([doc_output], tmpdir)

            # Validate with wrong template
            is_valid = validate_vlm_json_structure(json_path, "unknown_template")
            assert is_valid is False
