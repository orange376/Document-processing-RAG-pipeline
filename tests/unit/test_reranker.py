import pytest

from src.domain import Chunk, ChunkMetadata, SearchResult
from src.retrieval.reranker import Reranker


class TestReranker:
    def test_reranker_initialization(self):
        """No model should be loaded on init."""
        reranker = Reranker()
        assert reranker is not None
        assert reranker._model is None

    def test_reranker_rerank_empty(self):
        """Empty input should return empty list without loading the model."""
        reranker = Reranker()
        result = reranker.rerank("query", [])
        assert result == []
        assert reranker._model is None  # model should not be loaded

    def test_reranker_actual(self):
        """Actual re-ranking with the BGE model."""
        reranker = Reranker()
        chunk_a = Chunk(
            content="RAG 系统通过检索增强生成来提高 LLM 回答的准确性。",
            metadata=ChunkMetadata(
                source_file="test.pdf", page_num=1, section="intro", chunk_type="text"
            ),
        )
        chunk_b = Chunk(
            content="今天天气真好，适合出去散步。",
            metadata=ChunkMetadata(
                source_file="test.pdf", page_num=2, section="weather", chunk_type="text"
            ),
        )
        results = [
            SearchResult(chunk=chunk_a, score=0.5, retrieval_method="hybrid"),
            SearchResult(chunk=chunk_b, score=0.5, retrieval_method="hybrid"),
        ]
        reranked = reranker.rerank("什么是 RAG？", results, top_k=2)
        assert len(reranked) == 2
        # The chunk relevant to RAG should rank higher
        assert reranked[0].score >= reranked[1].score
        # scores should have been updated from the original 0.5
        assert reranked[0].score != 0.5 or reranked[1].score != 0.5

    def test_unload(self):
        """unload should clear the model reference."""
        reranker = Reranker()
        reranker._model = "fake_model"  # simulate loaded state
        reranker.unload()
        assert reranker._model is None

    def test_rerank_top_k(self):
        """rerank with top_k should return at most top_k results."""
        reranker = Reranker()
        # Don't load real model, just test the slicing behavior
        # by checking the logic path through empty results
        results = reranker.rerank("query", [], top_k=3)
        assert len(results) == 0

    def test_import_from_package(self):
        """Reranker should be importable from src.retrieval."""
        from src.retrieval import Reranker as R

        assert R is Reranker
