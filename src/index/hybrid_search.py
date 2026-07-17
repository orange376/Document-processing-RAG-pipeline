"""Hybrid search — RRF fusion of vector search and BM25 keyword search."""

from __future__ import annotations

from src.domain import Chunk, SearchResult

from .bm25_index import BM25Index
from .vector_store import VectorStore

_DEFAULT_RRF_K: int = 60
"""Default k constant for the RRF formula."""


class HybridSearch:
    """Hybrid search engine that fuses vector and BM25 results via RRF.

    Uses Reciprocal Rank Fusion (RRF) to combine ranked lists from
    semantic (vector) and keyword (BM25) retrieval, producing a single
    scored and deduplicated result list.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
    ):
        self._vector_store = vector_store or VectorStore()
        self._bm25 = bm25_index or BM25Index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 30,
    ) -> list[SearchResult]:
        """Hybrid search using RRF fusion.

        Parameters
        ----------
        query:
            Raw query text (passed to BM25 search).
        query_embedding:
            Dense vector representation of the query (passed to vector search).
        top_k:
            Number of results to return (default 30).

        Returns
        -------
        list[SearchResult]
            Top-*top_k* results sorted by RRF score descending.
        """
        # 1. Retrieve from both indexes
        vector_chunks: list[Chunk] = self._vector_store.search(
            query_embedding, top_k=top_k
        )
        bm25_results: list[SearchResult] = self._bm25.search(query, top_k=top_k)

        # 2. Convert vector chunks to SearchResult objects
        vector_results = [
            SearchResult(chunk=c, score=1.0, retrieval_method="vector")
            for c in vector_chunks
        ]

        # 3. Fuse via RRF
        fused = self._rrf_fusion(vector_results, bm25_results, k=_DEFAULT_RRF_K)

        # 4. Return top_k
        return fused[:top_k]

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fusion(
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        k: int = _DEFAULT_RRF_K,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion.

        Parameters
        ----------
        vector_results:
            Ranked results from vector search.
        bm25_results:
            Ranked results from BM25 search.
        k:
            RRF constant (default 60).

        Returns
        -------
        list[SearchResult]
            Results sorted by RRF score descending.
        """
        if not vector_results and not bm25_results:
            return []

        # Build rank dictionaries: chunk_id -> 1-based rank
        vector_ranks: dict[str, int] = {
            r.chunk.chunk_id: i + 1
            for i, r in enumerate(vector_results)
        }
        bm25_ranks: dict[str, int] = {
            r.chunk.chunk_id: i + 1
            for i, r in enumerate(bm25_results)
        }

        # Collect all unique chunk IDs
        all_ids: set[str] = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Build a lookup from chunk_id -> Chunk object (first source wins)
        chunk_map: dict[str, Chunk] = {}
        for r in vector_results:
            chunk_map[r.chunk.chunk_id] = r.chunk
        for r in bm25_results:
            # BM25 results may come after vector, but only fill in gaps
            if r.chunk.chunk_id not in chunk_map:
                chunk_map[r.chunk.chunk_id] = r.chunk

        # Compute RRF score for each unique document
        fused: list[SearchResult] = []
        for cid in all_ids:
            rrf_score = 0.0
            if cid in vector_ranks:
                rrf_score += 1.0 / (k + vector_ranks[cid])
            if cid in bm25_ranks:
                rrf_score += 1.0 / (k + bm25_ranks[cid])

            fused.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=rrf_score,
                    retrieval_method="hybrid",
                )
            )

        # Sort by RRF score descending
        fused.sort(key=lambda r: r.score, reverse=True)
        return fused
