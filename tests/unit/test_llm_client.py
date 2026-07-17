import pytest
from src.generation.llm_client import LLMClient


def _fake_settings(**overrides):
    """Create a fake settings object with default values."""
    defaults = dict(
        deepseek_api_key="test-key",
        deepseek_api_base="https://api.deepseek.com/v1",
        deepseek_model="deepseek-chat",
        qwen_api_key="",
        qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        qwen_model="qwen3-7b-plus",
        qwen_vl_model="qwen2.5-vl-3b-instruct",
    )
    defaults.update(overrides)
    return type("FakeSettings", (), defaults)()


class TestLLMClient:
    def test_llm_client_init(self):
        client = LLMClient(api_key="test-key")
        assert client is not None

    def test_llm_client_chat_empty_key(self, monkeypatch):
        # Force empty deepseek_api_key so api_key="" stays ""
        monkeypatch.setattr(
            "src.generation.llm_client.get_settings",
            lambda: _fake_settings(deepseek_api_key=""),
        )
        client = LLMClient(api_key="")
        result = client.chat("hello")
        assert result == ""

    def test_llm_client_chat_with_mock(self, httpx_mock):
        httpx_mock.add_response(
            json={"choices": [{"message": {"content": "test answer"}}]}
        )
        client = LLMClient(api_key="test-key")
        result = client.chat("hello")
        assert "test answer" in result

    def test_llm_client_chat_http_error_returns_empty(self, httpx_mock):
        httpx_mock.add_response(status_code=500)
        client = LLMClient(api_key="test-key")
        result = client.chat("hello")
        assert result == ""

    def test_import_from_package(self):
        from src.generation import LLMClient as LC
        assert LC is LLMClient

    def test_provider_deepseek_default(self, monkeypatch):
        # Use clean defaults so .env override doesn't affect assertion
        monkeypatch.setattr(
            "src.generation.llm_client.get_settings",
            lambda: _fake_settings(),
        )
        client = LLMClient(api_key="test-key")
        assert client._provider == "deepseek"
        assert client._base == "https://api.deepseek.com/v1"
        assert client._model == "deepseek-chat"

    def test_provider_qwen(self, monkeypatch):
        monkeypatch.setattr(
            "src.generation.llm_client.get_settings",
            lambda: _fake_settings(qwen_api_key="test-key"),
        )
        client = LLMClient(api_key="test-key", provider="qwen")
        assert client._provider == "qwen"
        assert "dashscope" in client._base
        assert client._model == "qwen3-7b-plus"
