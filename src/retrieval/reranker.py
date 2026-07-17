from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.domain.chunk import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """BGE-Reranker wrapper with lazy model loading (direct transformers)."""

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from src.config import get_settings

        model_dir: Path = get_settings().resolved_model_dir / "bge-reranker"
        if not model_dir.exists():
            logger.warning("Reranker model not found at %s", model_dir)
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading reranker model from %s", model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), torch_dtype="auto"
        )
        self._model.eval()

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 10
    ) -> list[SearchResult]:
        """Re-rank search results using the BGE reranker model."""
        if not results:
            return []
        self._lazy_load()
        if self._model is None:
            return results[:top_k]

        import torch

        pairs = [[query, r.chunk.content] for r in results]
        inputs = self._tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            # BGE reranker outputs raw logits — apply sigmoid to get 0-1 score
            logits = outputs.logits.squeeze(-1)
            scores = torch.sigmoid(logits).tolist()

        if isinstance(scores, float):
            scores = [scores]

        for r, s in zip(results, scores):
            r.score = float(s)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def unload(self):
        """Unload the model from memory."""
        self._model = None
        self._tokenizer = None
        import gc

        gc.collect()
