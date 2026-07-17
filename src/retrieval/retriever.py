"""Retriever — unified retrieval pipeline orchestrating hybrid search + reranker.

Usage::

    retriever = Retriever()
    results = retriever.retrieve("query text", embedding=[0.1] * 1024, top_k=10)
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

        Returns
        -------
        list[SearchResult]
            Ranked results — empty if no candidates found or on failure.
        """
        # Step 1: Hybrid search for initial candidates
        try:
            candidates: list[SearchResult] = self._hybrid.search(
                query, embedding, top_k=30
            )
        except Exception:
            logger.exception("Hybrid search failed, returning empty results")
            return []

        if not candidates:
            logger.info("Hybrid search returned no candidates")
            return []

        # Step 2: Reranker refines the candidates
        try:
            reranked: list[SearchResult] = self._reranker.rerank(
                query, candidates, top_k=top_k
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
