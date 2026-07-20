from __future__ import annotations

from pathlib import Path
from src.config import get_settings
from src.domain import Chunk


# Module-level singleton for the underlying QdrantClient.
# Local-mode Qdrant uses file locking and only allows one client per storage
# directory — sharing the instance across all VectorStore consumers avoids
# "already accessed by another instance" errors.
_QDRANT_CLIENT: object | None = None


def _get_qdrant_client(db_path: str) -> object:
    """Return the shared QdrantClient singleton, creating it on first call."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams, Distance

    Path(db_path).mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=db_path)

    # Check collection exists; create if not
    existing = client.get_collections()
    names = [c.name for c in existing.collections]

    collection_name = "documents"
    if collection_name not in names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024,  # bge-large-zh-v1.5 输出 1024 维
                distance=Distance.COSINE,
            ),
        )

    _QDRANT_CLIENT = client
    return client


class VectorStore:
    """向量存储 — 基于 Qdrant 本地模式

    The underlying ``QdrantClient`` is a module-level singleton so all
    consumers within the process share one client — vital for local-mode
    Qdrant which uses file-level locking.
    """

    def __init__(self, collection_name: str = "documents"):
        self._settings = get_settings()
        self._collection_name = collection_name
        self._client = None

    def _lazy_init(self):
        if self._client is not None:
            return
        db_path = str(self._settings.resolved_vector_db_dir)
        self._client = _get_qdrant_client(db_path)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """将 Chunk 列表写入向量库"""
        self._lazy_init()

        from qdrant_client.http.models import PointStruct

        points = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            points.append(PointStruct(
                id=hash(chunk.chunk_id) % (2**63),
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content[:1000],
                    "source_file": chunk.metadata.source_file if chunk.metadata else "",
                    "page_num": chunk.metadata.page_num if chunk.metadata else 0,
                    "section": chunk.metadata.section if chunk.metadata else "",
                    "chunk_type": chunk.metadata.chunk_type if chunk.metadata else "text",
                },
            ))

        if points:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
        return len(points)

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[Chunk]:
        """向量检索"""
        self._lazy_init()

        resp = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            with_vectors=True,
        )

        results = []
        for hit in (resp.points or []):
            payload = hit.payload or {}
            meta = None
            if payload.get("source_file"):
                from src.domain import ChunkMetadata
                meta = ChunkMetadata(
                    source_file=payload.get("source_file", ""),
                    page_num=payload.get("page_num", 0),
                    section=payload.get("section", ""),
                    chunk_type=payload.get("chunk_type", "text"),
                )

            results.append(Chunk(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                metadata=meta,
                embedding=hit.vector or [],
            ))

        return results

    def count(self) -> int:
        self._lazy_init()
        result = self._client.count(
            collection_name=self._collection_name,
            exact=True,
        )
        return result.count or 0

    def delete_by_source_file(self, source_file: str) -> int:
        """Delete all points whose payload matches *source_file*.

        Returns the number of points deleted (0 if the collection doesn't
        exist yet or nothing matched).
        """
        self._lazy_init()
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        try:
            result = self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_file),
                        )
                    ]
                ),
                wait=True,
            )
            return getattr(result, "count", 0)
        except Exception:
            return 0


def create_vector_store() -> VectorStore:
    return VectorStore()
