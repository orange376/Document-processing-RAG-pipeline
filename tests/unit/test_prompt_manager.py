import pytest
from src.generation.prompt_manager import PromptManager


class TestPromptManager:
    def test_render_qa_prompt(self):
        pm = PromptManager()
        result = pm.render("qa", query="test question", context="test context")
        assert "test question" in result
        assert "test context" in result

    def test_render_rewrite_prompt(self):
        pm = PromptManager()
        result = pm.render("rewrite", query="test question")
        assert "test question" in result

    def test_unknown_template_raises_value_error(self):
        pm = PromptManager()
        with pytest.raises(ValueError, match="Unknown template: nonexistent"):
            pm.render("nonexistent")

    def test_templates_contain_expected_keys(self):
        assert "qa" in PromptManager.TEMPLATES
        assert "rewrite" in PromptManager.TEMPLATES

    def test_import_from_package(self):
        from src.generation import PromptManager as PM
        assert PM is PromptManager

    def test_missing_kwargs_raises_key_error(self):
        pm = PromptManager()
        with pytest.raises(KeyError):
            pm.render("qa", query="only query")  # missing 'context'
