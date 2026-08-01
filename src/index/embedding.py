from __future__ import annotations

import logging
import threading

from src.config import get_settings
from src.domain import Chunk

logger = logging.getLogger(__name__)

# Module-level singleton for the underlying SentenceTransformer.
# Loaded once on first use and shared across all EmbeddingEngine instances.
_MODEL: object | None = None

# Serializes lazy singleton construction so concurrent asyncio.to_thread calls
# can't both build the model (check-then-act race).
_LOAD_LOCK = threading.Lock()


def _get_model(cache_dir: str) -> object:
    """Return the shared SentenceTransformer singleton (bge-base-zh, ONNX backend).

    Tries ONNX runtime first for CPU speed; falls back to torch backend
    (same 768-dim model) if ONNX export is unavailable.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is None:
            from sentence_transformers import SentenceTransformer

            try:
                _MODEL = SentenceTransformer(
                    "BAAI/bge-base-zh-v1.5",
                    cache_folder=cache_dir,
                    backend="onnx",
                    device="cpu",
                )
            except Exception:
                logger.warning("ONNX backend unavailable, falling back to torch", exc_info=True)
                # ONNX 不可用时回退 torch 后端（同模型同维度，仅更慢）
                _MODEL = SentenceTransformer(
                    "BAAI/bge-base-zh-v1.5",
                    cache_folder=cache_dir,
                    device="cpu",
                )
    return _MODEL


class EmbeddingEngine:
    """Embedding 引擎 — 基于 bge-base-zh-v1.5 (768 维, ONNX 加速)

    The underlying SentenceTransformer is a module-level singleton so it
    survives across requests without reloading.
    """

    def __init__(self):
        self._settings = get_settings()

    def unload(self):
        """No-op — the singleton model stays loaded across requests."""
        pass

    def embed(self, text: str) -> list[float]:
        """对单段文本编码"""
        model = _get_model(str(self._settings.resolved_model_dir / "bge-base-zh"))
        emb = model.encode(text, show_progress_bar=False)
        return emb.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        model = _get_model(str(self._settings.resolved_model_dir / "bge-base-zh"))
        embs = model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embs]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """为 Chunk 列表填充 embedding（优先走 Redis 缓存）。"""
        from src.cache import RedisCache
        cache = RedisCache()

        uncached: list[tuple[int, Chunk]] = []
        embed_dim = self._settings.embedding_dim
        for i, chunk in enumerate(chunks):
            cached_vec = cache.get_embedding(chunk.content)
            # Accept a cache hit only if the vector matches the current model
            # dimension — stale entries from an older (e.g. 1024-dim) model
            # would otherwise fail Qdrant dimension validation at index time.
            if cached_vec is not None and len(cached_vec) == embed_dim:
                chunk.embedding = cached_vec
            else:
                uncached.append((i, chunk))

        if not uncached:
            return chunks

        texts = [c.content for _, c in uncached]
        embeddings = self.embed_batch(texts)
        for (idx, chunk), emb in zip(uncached, embeddings):
            chunk.embedding = emb
            cache.set_embedding(chunk.content, emb)

        return chunks


def create_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()
