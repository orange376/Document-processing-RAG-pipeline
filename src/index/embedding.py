from __future__ import annotations

import logging
import threading
from pathlib import Path

from src.config import get_settings
from src.domain import Chunk

logger = logging.getLogger(__name__)

# Module-level singleton for the underlying SentenceTransformer.
# Loaded once on first use and shared across all EmbeddingEngine instances.
_MODEL: object | None = None

# Serializes lazy singleton construction so concurrent asyncio.to_thread calls
# can't both build the model (check-then-act race).
_LOAD_LOCK = threading.Lock()


def _resolve_model_dir(cache_dir: str) -> Path | None:
    """Resolve the local snapshot directory for the embedding model.

    Returns ``None`` if the model isn't cached locally yet (first run) so the
    caller can fall back to hub loading (which downloads it).
    """
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                "BAAI/bge-base-zh-v1.5",
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )
    except Exception:
        logger.warning("本地模型目录不可用，将回退 hub 加载", exc_info=True)
        return None


def _persist_onnx_export(model: object, model_dir: Path) -> None:
    """Persist the exported ONNX model so future startups skip the ~26s re-export.

    sentence-transformers 5.x exports to ONNX in-memory on first load and only
    writes ``onnx/model.onnx`` if we call ``save_pretrained()`` ourselves. When
    the model is loaded from a *local* path, ST checks that directory for
    ``*.onnx`` and skips export if present — so saving here means the conversion
    is paid once, not on every server start.
    """
    try:
        # Match ST's backend_should_export check exactly: it looks for
        # ``model.onnx`` at the root or ``onnx/model.onnx`` in the subfolder.
        # A loose ``**/*.onnx`` glob would wrongly match stray onnx files
        # elsewhere (e.g. a backup dir) and skip the save.
        if (model_dir / "onnx" / "model.onnx").exists():
            return  # already persisted on a previous run
        model.save_pretrained(str(model_dir))
        logger.info("ONNX 导出已持久化到 %s/onnx/", model_dir)
    except Exception:
        # Non-fatal: the in-memory model still works this run, we just pay the
        # export cost again on the next process start.
        logger.warning("ONNX 导出持久化失败（本次运行不受影响）", exc_info=True)


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

            # Load from the resolved local snapshot path, not the hub id: ST's
            # ``backend_should_export`` globs the load path for ``*.onnx`` when
            # it's a local dir, but checks the remote repo when it's a hub id —
            # so only a local path can skip the ONNX export on the next start.
            model_dir = _resolve_model_dir(cache_dir)
            onnx_kwargs = (
                {"cache_folder": cache_dir} if model_dir is None else {}
            )
            model_arg = str(model_dir) if model_dir is not None else "BAAI/bge-base-zh-v1.5"

            try:
                _MODEL = SentenceTransformer(
                    model_arg,
                    backend="onnx",
                    device="cpu",
                    **onnx_kwargs,
                )
            except Exception:
                logger.warning("ONNX backend unavailable, falling back to torch", exc_info=True)
                # ONNX 不可用时回退 torch 后端（同模型同维度，仅更慢）
                _MODEL = SentenceTransformer(
                    model_arg,
                    cache_folder=cache_dir,
                    device="cpu",
                )
            else:
                # Persist the export so the ~26s conversion is paid once, not
                # on every server start (only meaningful when loading locally).
                if model_dir is not None:
                    _persist_onnx_export(_MODEL, model_dir)
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
