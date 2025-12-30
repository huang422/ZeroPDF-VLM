"""Unit tests for field schema validation and template definitions."""

import pytest
from vlm_pdf_recognizer.recognition.field_schema import (
    FieldSchema,
    TemplateSchema,
    TEMPLATE_SCHEMAS,
    PROMPT_TEMPLATES
)


class TestFieldSchema:
    """Test FieldSchema dataclass and validation."""

    def test_title_field_validation(self):
        """Title fields must have predefined_value."""
        field = FieldSchema(
            field_id="test_title",
            field_type="title",
            template_id="test",
            description="Test title",
            prompt_template_key="",
            predefined_value="Test Document"
        )
        field.validate()  # Should not raise

    def test_non_title_field_validation(self):
        """Non-title fields must NOT have predefined_value."""
        field = FieldSchema(
            field_id="test_checkbox",
            field_type="checkbox",
            template_id="test",
            description="Test checkbox",
            prompt_template_key="checkbox",
            predefined_value=None
        )
        field.validate()  # Should not raise

    def test_title_field_without_predefined_value_fails(self):
        """Title fields without predefined_value should fail validation."""
        field = FieldSchema(
            field_id="test_title",
            field_type="title",
            template_id="test",
            description="Test title",
            prompt_template_key="",
            predefined_value=None
        )
        with pytest.raises(AssertionError, match="must have predefined_value"):
            field.validate()

    def test_non_title_field_with_predefined_value_fails(self):
        """Non-title fields with predefined_value should fail validation."""
        field = FieldSchema(
            field_id="test_text",
            field_type="text",
            template_id="test",
            description="Test text",
            prompt_template_key="text",
            predefined_value="Invalid"
        )
        with pytest.raises(AssertionError, match="must have null predefined_value"):
            field.validate()

    def test_get_prompt_for_title_returns_empty(self):
        """Title fields should return empty prompt."""
        field = FieldSchema(
            field_id="test_title",
            field_type="title",
            template_id="test",
            description="Test title",
            prompt_template_key="",
            predefined_value="Test"
        )
        assert field.get_prompt(PROMPT_TEMPLATES) == ""

    def test_get_prompt_for_checkbox(self):
        """Checkbox fields should return checkbox prompt."""
        field = FieldSchema(
            field_id="test_checkbox",
            field_type="checkbox",
            template_id="test",
            description="Test checkbox",
            prompt_template_key="checkbox",
            predefined_value=None
        )
        prompt = field.get_prompt(PROMPT_TEMPLATES)
        assert "<image>" in prompt
        assert "has_content" in prompt


class TestTemplateSchema:
    """Test TemplateSchema dataclass and validation."""

    def test_contractor_1_template_validation(self):
        """contractor_1 template should validate successfully."""
        schema = TEMPLATE_SCHEMAS["contractor_1"]
        schema.validate()  # Should not raise

    def test_contractor_2_template_validation(self):
        """contractor_2 template should validate successfully."""
        schema = TEMPLATE_SCHEMAS["contractor_2"]
        schema.validate()  # Should not raise

    def test_enterprise_1_template_validation(self):
        """enterprise_1 template should validate successfully."""
        schema = TEMPLATE_SCHEMAS["enterprise_1"]
        schema.validate()  # Should not raise

    def test_contractor_1_field_count(self):
        """contractor_1 should have exactly 13 fields."""
        schema = TEMPLATE_SCHEMAS["contractor_1"]
        assert schema.field_count == 13

    def test_contractor_2_field_count(self):
        """contractor_2 should have exactly 2 fields."""
        schema = TEMPLATE_SCHEMAS["contractor_2"]
        assert schema.field_count == 2

    def test_enterprise_1_field_count(self):
        """enterprise_1 should have exactly 14 fields."""
        schema = TEMPLATE_SCHEMAS["enterprise_1"]
        assert schema.field_count == 14

    def test_get_field_by_id(self):
        """Should retrieve field by ID."""
        schema = TEMPLATE_SCHEMAS["contractor_1"]
        field = schema.get_field_by_id("VX1")
        assert field is not None
        assert field.field_id == "VX1"
        assert field.field_type == "checkbox"

    def test_get_nonexistent_field_returns_none(self):
        """Should return None for nonexistent field ID."""
        schema = TEMPLATE_SCHEMAS["contractor_1"]
        field = schema.get_field_by_id("nonexistent")
        assert field is None


class TestPromptTemplates:
    """Test prompt template definitions."""

    def test_all_prompt_types_present(self):
        """All required prompt types should be defined."""
        required_types = ["checkbox", "stamp", "text", "number", "generic"]
        for prompt_type in required_types:
            assert prompt_type in PROMPT_TEMPLATES

    def test_prompts_have_image_prefix(self):
        """All prompts (except generic fallback) should start with <image>."""
        for prompt_type, prompt in PROMPT_TEMPLATES.items():
            if prompt:  # Skip empty prompts
                assert prompt.startswith("<image>"), f"Prompt '{prompt_type}' missing <image> prefix"

    def test_prompts_have_json_instruction(self):
        """All prompts should include JSON output instruction."""
        for prompt_type, prompt in PROMPT_TEMPLATES.items():
            if prompt:  # Skip empty prompts
                assert "has_content" in prompt, f"Prompt '{prompt_type}' missing has_content instruction"


class TestTemplateIntegrity:
    """Test integrity of template-to-field mappings."""

    def test_vx1_field_in_contractor_1(self):
        """contractor_1 should have VX1 checkbox field."""
        schema = TEMPLATE_SCHEMAS["contractor_1"]
        vx1 = schema.get_field_by_id("VX1")
        assert vx1 is not None
        assert vx1.field_type == "checkbox"

    def test_vx1_field_in_enterprise_1(self):
        """enterprise_1 should have VX1 checkbox field."""
        schema = TEMPLATE_SCHEMAS["enterprise_1"]
        vx1 = schema.get_field_by_id("VX1")
        assert vx1 is not None
        assert vx1.field_type == "checkbox"

    def test_date_fields_in_contractor_1(self):
        """contractor_1 should have year, month, date fields."""
        schema = TEMPLATE_SCHEMAS["contractor_1"]
        for field_id in ["year", "month", "date"]:
            field = schema.get_field_by_id(field_id)
            assert field is not None
            assert field.field_type == "number"

    def test_title_fields_have_correct_values(self):
        """All title fields should have correct predefined values."""
        expected_titles = {
            "contractor_1": "企業負責人電信信評報告之使用授權書",
            "contractor_2": "個人資料特定目的外用告知事項暨同意書",
            "enterprise_1": "企業電信信評報告之使用授權書"
        }

        for template_id, expected_title in expected_titles.items():
            schema = TEMPLATE_SCHEMAS[template_id]
            title_field = schema.get_field_by_id(schema.title_field_id)
            assert title_field is not None
            assert title_field.field_type == "title"
            assert title_field.predefined_value == expected_title
