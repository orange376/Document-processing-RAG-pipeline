from __future__ import annotations

from typing import Any

from src.domain import Chunk, LayoutElement, Table


class ConfidenceScorer:
    """5-dimension confidence scorer for RAG pipeline quality evaluation.

    Scoring dimensions and weights:
        layout_quality  (0.25): mean confidence of layout elements; 0 if no elements
        ocr_confidence  (0.20): mean OCR confidence; 0 if empty
        table_integrity (0.15): 1.0 if all tables are well-formed, 0.5 if some issues,
                                0.0 if empty
        chunk_coherence (0.20): ratio of chunks with non-empty content and metadata
        reranker_score  (0.20): max reranker score; 0 if empty
    """

    WEIGHTS: dict[str, float] = {
        "layout_quality": 0.25,
        "ocr_confidence": 0.20,
        "table_integrity": 0.15,
        "chunk_coherence": 0.20,
        "reranker_score": 0.20,
    }

    def score(
        self,
        layout_elements: list[LayoutElement] | None = None,
        ocr_results: list[Any] | None = None,
        tables: list[Table] | None = None,
        chunks: list[Chunk] | None = None,
        reranker_scores: list[float] | None = None,
        num_results: int | None = None,
    ) -> dict[str, Any]:
        """Compute overall confidence score and per-dimension details.

        Args:
            layout_elements: List of LayoutElement objects from layout analysis.
            ocr_results: List of OCR result objects with a ``confidence`` attribute,
                         or tuples of (bbox, text, confidence).
            tables: List of Table objects.
            chunks: List of Chunk objects.
            reranker_scores: List of reranker score floats.
            num_results: When provided (query-time scoring), switches to
                         query-appropriate weights (reranker_score=0.6,
                         result_coverage=0.4) that sum to 1.0, making the
                         score usable even when document-level dimensions
                         are unavailable.

        Returns:
            dict with ``overall`` (float, 0-1) and ``details`` (dict of str->float).
        """
        details = {
            "layout_quality": self._score_layout(layout_elements or []),
            "ocr_confidence": self._score_ocr(ocr_results or []),
            "table_integrity": self._score_tables(tables or []),
            "chunk_coherence": self._score_chunks(chunks or []),
            "reranker_score": self._score_reranker(reranker_scores or []),
        }

        if num_results is not None:
            # Query-time scoring — only reranker_score and result_coverage
            # are meaningful; switch to query-appropriate weights.
            details["result_coverage"] = min(num_results / 10.0, 1.0)
            weights: dict[str, float] = {
                "reranker_score": 0.6,
                "result_coverage": 0.4,
            }
        else:
            weights = self.WEIGHTS

        overall = sum(details[k] * weights[k] for k in weights)
        return {"overall": round(overall, 4), "details": details}

    # ------------------------------------------------------------------
    # Private dimension scorers
    # ------------------------------------------------------------------

    def _score_layout(self, layout_elements: list[LayoutElement]) -> float:
        """Mean confidence of layout elements; 0 if no elements."""
        if not layout_elements:
            return 0.0
        confidences = [el.confidence for el in layout_elements]
        return sum(confidences) / len(confidences)

    def _score_ocr(self, ocr_results: list[Any]) -> float:
        """Mean OCR confidence; 0 if empty."""
        if not ocr_results:
            return 0.0

        confidences: list[float] = []
        for item in ocr_results:
            if hasattr(item, "confidence"):
                confidences.append(item.confidence)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                # (bbox, text, confidence) tuple from easyocr
                try:
                    confidences.append(float(item[2]))
                except (TypeError, ValueError):
                    continue

        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    def _score_tables(self, tables: list[Table]) -> float:
        """Evaluate table integrity.

        Returns:
            1.0 if all tables are well-formed (rows >= 1 and cols >= 1).
            0.5 if at least one table but some have issues.
            0.0 if no tables.
        """
        if not tables:
            return 0.0

        well_formed = sum(
            1 for t in tables if t.num_rows >= 1 and t.num_cols >= 1
        )
        total = len(tables)

        if well_formed == total:
            return 1.0
        elif well_formed > 0:
            return 0.5
        else:
            return 0.0

    def _score_chunks(self, chunks: list[Chunk]) -> float:
        """Ratio of chunks that have non-empty content and metadata."""
        if not chunks:
            return 0.0

        good = sum(
            1 for c in chunks if c.content and c.metadata is not None
        )
        return good / len(chunks)

    def _score_reranker(self, reranker_scores: list[float]) -> float:
        """Max reranker score; 0 if empty."""
        if not reranker_scores:
            return 0.0
        return max(reranker_scores)
