from __future__ import annotations

from src.config import get_settings
from src.domain import Chunk


class EmbeddingEngine:
    """Embedding 引擎 — 基于 bge-large-zh-v1.5"""

    def __init__(self):
        self._model = None
        self._settings = get_settings()

    def _lazy_load(self):
        if self._model is None:
            from FlagEmbedding import FlagModel
            import torch
            use_fp16 = torch.cuda.is_available()
            self._model = FlagModel(
                'BAAI/bge-large-zh-v1.5',
                use_fp16=use_fp16,
                cache_folder=str(self._settings.resolved_model_dir / "bge-large-zh"),
            )

    def unload(self):
        self._model = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed(self, text: str) -> list[float]:
        """对单段文本编码"""
        self._lazy_load()
        emb = self._model.encode([text])
        return emb[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码"""
        self._lazy_load()
        embs = self._model.encode(texts)
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
