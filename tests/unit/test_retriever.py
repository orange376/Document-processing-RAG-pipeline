"""Retriever 单元测试"""

from unittest.mock import MagicMock, PropertyMock

import pytest

from src.domain import Chunk, ChunkMetadata, SearchResult
from src.retrieval.retriever import Retriever


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_chunk(chunk_id: str, content: str = "dummy") -> Chunk:
    """Quick Chunk builder."""
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            source_file="doc.pdf",
            page_num=1,
            section="test",
            chunk_type="text",
        ),
    )


def make_result(chunk_id: str, score: float = 1.0, method: str = "hybrid") -> SearchResult:
    """Quick SearchResult builder."""
    return SearchResult(
        chunk=make_chunk(chunk_id),
        score=score,
        retrieval_method=method,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestRetrieverInitialization:
    """Retriever 初始化测试"""

    def test_init_default(self):
        """使用默认参数初始化应成功创建实例。"""
        retriever = Retriever()
        assert retriever is not None
        assert retriever._vector_store is not None
        assert retriever._bm25 is not None
        assert retriever._hybrid is not None
        assert retriever._reranker is not None

    def test_init_with_explicit_components(self):
        """传入显式子组件实例应被正确使用。"""
        mock_vs = MagicMock()
        mock_bm25 = MagicMock()
        mock_hybrid = MagicMock()
        mock_reranker = MagicMock()

        retriever = Retriever(
            vector_store=mock_vs,
            bm25_index=mock_bm25,
            hybrid_search=mock_hybrid,
            reranker=mock_reranker,
        )

        assert retriever._vector_store is mock_vs
        assert retriever._bm25 is mock_bm25
        assert retriever._hybrid is mock_hybrid
        assert retriever._reranker is mock_reranker

    def test_retriever_importable(self):
        """Retriever 可从 src.retrieval 导入。"""
        from src.retrieval import Retriever as R
        assert R is Retriever


class TestRetrieverRetrieve:
    """Retriever.retrieve 方法测试"""

    def test_retrieve_empty_candidates(self):
        """混合检索返回空结果时，retrieve 应返回空列表。"""
        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = []
        mock_reranker = MagicMock()

        retriever = Retriever(hybrid_search=mock_hybrid, reranker=mock_reranker)
        results = retriever.retrieve("test", [0.1] * 1024, top_k=5)

        assert results == []
        # Reranker 不应被调用
        mock_reranker.rerank.assert_not_called()

    def test_retrieve_successful_flow(self):
        """正常流程：混合搜索 -> Reranker 精排。"""
        candidates = [
            make_result("a", score=0.8, method="hybrid"),
            make_result("b", score=0.6, method="hybrid"),
        ]
        reranked = [
            make_result("b", score=0.9, method="hybrid"),
            make_result("a", score=0.7, method="hybrid"),
        ]

        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = candidates
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = reranked

        retriever = Retriever(hybrid_search=mock_hybrid, reranker=mock_reranker)
        results = retriever.retrieve("query", [0.1] * 1024, top_k=2)

        assert results == reranked
        mock_hybrid.search.assert_called_once_with("query", [0.1] * 1024, top_k=30)
        mock_reranker.rerank.assert_called_once_with("query", candidates, top_k=2)

    def test_retrieve_top_k_default(self):
        """不传 top_k 时，默认值为 10。"""
        candidates = [make_result(str(i)) for i in range(20)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = candidates
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = candidates[:10]

        retriever = Retriever(hybrid_search=mock_hybrid, reranker=mock_reranker)
        results = retriever.retrieve("query", [0.1] * 1024)

        assert len(results) == 10
        mock_reranker.rerank.assert_called_once_with("query", candidates, top_k=10)


class TestRetrieverGracefulDegradation:
    """Retriever 容错降级测试"""

    def test_hybrid_search_failure_returns_empty(self):
        """混合搜索抛出异常时，retrieve 应返回空列表。"""
        mock_hybrid = MagicMock()
        mock_hybrid.search.side_effect = RuntimeError("Qdrant connection failed")
        mock_reranker = MagicMock()

        retriever = Retriever(hybrid_search=mock_hybrid, reranker=mock_reranker)
        results = retriever.retrieve("query", [0.1] * 1024)

        assert results == []
        mock_reranker.rerank.assert_not_called()

    def test_reranker_failure_falls_back_to_candidates(self):
        """Reranker 抛出异常时，回退到混合搜索结果并截断到 top_k。"""
        candidates = [make_result(str(i)) for i in range(15)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = candidates
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("OOM")

        retriever = Retriever(hybrid_search=mock_hybrid, reranker=mock_reranker)
        results = retriever.retrieve("query", [0.1] * 1024, top_k=5)

        # 应回退到 candidates 的前 5 条
        assert len(results) == 5
        assert results == candidates[:5]

    def test_unload_reranker(self):
        """unload 应尝试释放 reranker 模型。"""
        mock_reranker = MagicMock()
        retriever = Retriever(reranker=mock_reranker)
        retriever.unload()
        mock_reranker.unload.assert_called_once()

    def test_unload_reranker_failure(self):
        """unload 时 reranker 抛异常不应传播。"""
        mock_reranker = MagicMock()
        mock_reranker.unload.side_effect = RuntimeError("unload failed")
        retriever = Retriever(reranker=mock_reranker)
        # 不应抛出异常
        retriever.unload()
        mock_reranker.unload.assert_called_once()
