"""VectorStore collection 配置测试。"""

from src.config import get_settings
from src.index.vector_store import VectorStore, _ensure_collection


class TestVectorStoreConfig:
    def test_default_collection_from_settings(self):
        s = get_settings()
        vs = VectorStore()
        assert vs._collection_name == s.qdrant_collection == "documents_v2"
        assert vs._vector_size == s.embedding_dim == 768

    def test_ensure_collection_idempotent(self, tmp_path):
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(tmp_path))
        _ensure_collection(client, "test_col", 768)
        _ensure_collection(client, "test_col", 768)  # 二次调用不报错
        names = [c.name for c in client.get_collections().collections]
        assert "test_col" in names
