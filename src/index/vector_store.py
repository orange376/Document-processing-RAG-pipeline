from __future__ import annotations

from pathlib import Path
from src.config import get_settings
from src.domain import Chunk


class VectorStore:
    """向量存储 — 基于 Qdrant 本地模式"""

    def __init__(self, collection_name: str = "documents"):
        self._settings = get_settings()
        self._collection_name = collection_name
        self._client = None

    def _lazy_init(self):
        if self._client is not None:
            return

        from qdrant_client import QdrantClient
        from qdrant_client.http.models import VectorParams, Distance

        db_path = str(self._settings.resolved_vector_db_dir)
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._client = QdrantClient(path=db_path)

        # 检查 collection 是否存在，不存在则创建
        existing = self._client.get_collections()
        names = [c.name for c in existing.collections]

        if self._collection_name not in names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=1024,  # bge-large-zh-v1.5 输出 1024 维
                    distance=Distance.COSINE,
                ),
            )

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


def create_vector_store() -> VectorStore:
    return VectorStore()
