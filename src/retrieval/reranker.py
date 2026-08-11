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
        self._device = "cpu"

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

        # Move to GPU when available — CPU inference over 30 candidate pairs was
        # the dominant retrieval cost (~10s per query).
        try:
            import torch

            if torch.cuda.is_available():
                self._device = "cuda"
                self._model = self._model.to("cuda")
                logger.info("Reranker moved to CUDA")
        except Exception:
            logger.warning("Failed to move reranker to CUDA, using CPU", exc_info=True)
            self._device = "cpu"

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 10
    ) -> list[SearchResult]:
        """Re-rank search results using the BGE reranker model (with Redis cache)."""
        if not results:
            return []
        self._lazy_load()
        if self._model is None:
            return results[:top_k]

        from src.cache import RedisCache

        cache = RedisCache()

        # Phase 1: check cache for each result
        uncached_pairs: list[tuple[int, SearchResult]] = []
        for i, r in enumerate(results):
            cached = cache.get_rerank_score(query, r.chunk.chunk_id)
            if cached is not None:
                r.score = cached
            else:
                uncached_pairs.append((i, r))

        if uncached_pairs:
            import torch

            pairs = [[query, r.chunk.content] for _, r in uncached_pairs]
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            if self._device != "cpu":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits.squeeze(-1)
                if logits.dim() == 0:
                    scores_list = [float(torch.sigmoid(logits))]
                else:
                    scores_list = torch.sigmoid(logits).tolist()

            for (idx, r), s in zip(uncached_pairs, scores_list):
                r.score = float(s)
                cache.set_rerank_score(query, r.chunk.chunk_id, float(s))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def unload(self):
        """Unload the model from memory."""
        self._model = None
        self._tokenizer = None
        import gc

        gc.collect()
