import pytest
from src.index import EmbeddingEngine


class TestEmbeddingEngine:
    def test_initialization(self):
        engine = EmbeddingEngine()
        assert engine._model is None  # lazy

    def test_embed(self):
        engine = EmbeddingEngine()
        emb = engine.embed("测试文本")
        assert len(emb) == 1024  # bge-large-zh 维度

    def test_unload(self):
        engine = EmbeddingEngine()
        engine.unload()
        assert True
