from __future__ import annotations

from src.config import get_settings
from src.domain import Chunk


# Module-level singleton for the underlying FlagModel.
# Loaded once on first use and shared across all EmbeddingEngine instances
# so that we never pay the 3–10 s reload penalty per request.
_MODEL: object | None = None


def _get_model(cache_dir: str) -> object:
    """Return the shared FlagModel singleton, loading it on first call."""
    global _MODEL
    if _MODEL is None:
        from FlagEmbedding import FlagModel
        import torch

        use_fp16 = torch.cuda.is_available()
        _MODEL = FlagModel(
            "BAAI/bge-large-zh-v1.5",
            use_fp16=use_fp16,
            cache_folder=cache_dir,
        )
    return _MODEL


class EmbeddingEngine:
    """Embedding 引擎 — 基于 bge-large-zh-v1.5

    The underlying FlagModel is a module-level singleton so it survives
    across requests without reloading.
    """

    def __init__(self):
        self._settings = get_settings()

    def unload(self):
        """No-op — the singleton model stays loaded across requests.
        Kept for backward-compatibility with callers that call unload()."""
        pass

    def embed(self, text: str) -> list[float]:
        """对单段文本编码"""
        model = _get_model(str(self._settings.resolved_model_dir / "bge-large-zh"))
        emb = model.encode([text])
        return emb[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        model = _get_model(str(self._settings.resolved_model_dir / "bge-large-zh"))
        embs = model.encode(texts)
        return [e.tolist() for e in embs]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """为 Chunk 列表填充 embedding"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks


def create_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()
