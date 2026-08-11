"""Retriever — unified retrieval pipeline orchestrating hybrid search + reranker.

Usage::

    retriever = Retriever()
    results = retriever.retrieve("query text", embedding=[0.1] * 768, top_k=10)
"""

from __future__ import annotations

import logging
from typing import Any

from src.domain.chunk import SearchResult
from src.index.hybrid_search import HybridSearch
from src.index.bm25_index import BM25Index
from src.index.vector_store import VectorStore
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

# Function words that carry no domain signal for query expansion.
_STOPWORDS = frozenset(
    "的 了 是 在 和 与 及 或 等 中 上 下 为 对 以 将 从 到 通过 包括 以及 采用 "
    "提供 进行 可以 需要 支持 相关 一个 一种 这些 那些 该 此 这个 我们 本文 "
    "使用 具有 用于 主要 如下 说明 例如 其中 同时 并且 或者 如果 那么 由于 "
    "因此 根据 基于 方面 内容 对于 关于 一些 很多 非常 更加 来说 情况下 过程中 部分".split()
)


def _extract_domain_terms(texts: list[str], exclude: str) -> list[str]:
    """Extract frequent meaningful terms from *texts* via jieba, excluding those
    already present in *exclude* (typically the query).

    Used to expand the reranker query so bge-reranker-base — which is heavily
    lexical-overlap sensitive — can score relevant chunks highly even when the
    user's query is a question with little term overlap.
    """
    try:
        import jieba
        import jieba.analyse
    except Exception:
        return []

    try:
        freq: dict[str, int] = {}
        for text in texts:
            for w in jieba.cut(text):
                w = w.strip()
                if len(w) < 2 or w in _STOPWORDS or w in exclude:
                    continue
                freq[w] = freq.get(w, 0) + 1
        # Order by frequency, then prefer longer (more specific) terms on ties.
        return sorted(freq, key=lambda w: (freq[w], len(w)), reverse=True)[:10]
    except Exception:
        logger.warning("Query term extraction failed", exc_info=True)
        return []


class Retriever:
    """Main retrieval entry point — hybrid search followed by reranker."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker: Reranker | None = None,
        hybrid_search: HybridSearch | None = None,
    ):
        self._vector_store: VectorStore = vector_store or VectorStore()
        self._bm25: BM25Index = bm25_index or BM25Index()
        self._hybrid: HybridSearch = hybrid_search or HybridSearch(
            self._vector_store, self._bm25
        )
        self._reranker: Reranker = reranker or Reranker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 10,
        source_files: list[str] | None = None,
    ) -> list[SearchResult]:
        """Run the full retrieval pipeline.

        Parameters
        ----------
        query:
            Raw query text.
        embedding:
            Dense query embedding for vector search.
        top_k:
            Number of final results to return (default 10).
        source_files:
            Optional list of source filenames to restrict search scope.

        Returns
        -------
        list[SearchResult]
            Ranked results — empty if no candidates found or on failure.
        """
        # Step 1: Hybrid search for initial candidates
        try:
            candidates: list[SearchResult] = self._hybrid.search(
                query, embedding, top_k=30, source_files=source_files,
            )
        except Exception:
            logger.exception("Hybrid search failed, returning empty results")
            return []

        if not candidates:
            logger.info("Hybrid search returned no candidates")
            return []

        # Step 2: Reranker refines the candidates.
        # Enrich the reranker query with domain terms from the top candidates:
        # bge-reranker-base is lexical-overlap sensitive, so a question-form
        # query (or a generic rewrite) scores relevant chunks near 0 unless the
        # query shares their domain terms. Appending frequent terms from the
        # top hits makes the cross-encoder actually fire on the good matches.
        terms = _extract_domain_terms(
            [c.chunk.content for c in candidates[:5]], exclude=query
        )
        # Require a few real terms — a single stray token adds noise, not signal.
        rerank_query = f"{query} {' '.join(terms)}".strip() if len(terms) >= 3 else query
        try:
            reranked: list[SearchResult] = self._reranker.rerank(
                rerank_query, candidates, top_k=top_k
            )
        except Exception:
            logger.exception(
                "Reranker failed, falling back to hybrid search top-%d", top_k
            )
            return candidates[:top_k]

        return reranked

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self):
        """Release loaded models from memory."""
        try:
            self._reranker.unload()
        except Exception:
            logger.exception("Failed to unload reranker")
