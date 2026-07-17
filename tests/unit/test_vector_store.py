import pytest
from src.domain import Chunk, ChunkMetadata
from src.index import VectorStore


class TestVectorStore:
    def test_in_memory_mode(self):
        """测试内存模式（不用真实文件）"""
        import tempfile
        import os
        from qdrant_client import QdrantClient

        # 直接用临时路径测试 Qdrant 本地模式是否可用
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            client = QdrantClient(path=tmpdir)
            assert client is not None

    def test_create_store(self):
        store = VectorStore(collection_name="test_collection")
        # 仅测试初始化不报错
        assert store is not None
        assert store._client is None  # lazy
