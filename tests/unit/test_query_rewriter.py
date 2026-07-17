import pytest
from src.retrieval.query_rewriter import QueryRewriter


class TestQueryRewriter:
    def test_rewrite_returns_original_when_no_llm(self):
        rewriter = QueryRewriter()
        result = rewriter.rewrite("test query")
        assert result == "test query"

    def test_rewrite_returns_original_when_llm_raises_exception(self):
        class FailingLLM:
            def chat(self, prompt, system=""):
                raise RuntimeError("LLM unavailable")

        rewriter = QueryRewriter(llm_client=FailingLLM())
        result = rewriter.rewrite("test query")
        assert result == "test query"

    def test_import_from_package(self):
        from src.retrieval import QueryRewriter as QR
        assert QR is QueryRewriter
