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

    def test_rewrite_memoized_same_llm_called_once(self, monkeypatch):
        """相同 query 改写结果应记忆化（LLM 只调一次），保证下游缓存可命中。"""
        # 隔离真实 Redis（跨 pytest 运行会持久化改写结果，导致 LLM 不被调用）
        from src.cache.redis_cache import RedisCache

        monkeypatch.setattr(RedisCache, "get_rewrite", lambda self, q: None)
        monkeypatch.setattr(RedisCache, "set_rewrite", lambda self, q, r: None)

        calls = []

        class CountingLLM:
            def chat(self, prompt, system="", temperature=0.3):
                calls.append(prompt)
                return "改写结果"

        rewriter = QueryRewriter(llm_client=CountingLLM())
        assert rewriter.rewrite("同一个问题") == "改写结果"
        assert rewriter.rewrite("同一个问题") == "改写结果"
        assert rewriter.rewrite("同一个问题") == "改写结果"
        assert len(calls) == 1  # 只调了一次 LLM

    def test_rewrite_memo_distinct_per_query(self):
        """不同 query 不应共用同一改写缓存。"""
        class StubLLM:
            def chat(self, prompt, system="", temperature=0.3):
                return f"改写{len(prompt)}"

        rewriter = QueryRewriter(llm_client=StubLLM())
        assert rewriter.rewrite("短问题") != rewriter.rewrite("一个更长一些的问题")
