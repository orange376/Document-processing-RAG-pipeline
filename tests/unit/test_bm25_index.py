"""BM25Index 单元测试"""

from src.domain import Chunk, ChunkMetadata, SearchResult
from src.index.bm25_index import BM25Index


def make_metadata(
    source_file: str = "doc.pdf",
    page_num: int = 1,
    section: str = "test",
    chunk_type: str = "text",
) -> ChunkMetadata:
    """辅助：快速构建 ChunkMetadata。"""
    return ChunkMetadata(
        source_file=source_file,
        page_num=page_num,
        section=section,
        chunk_type=chunk_type,
    )


class TestBM25Index:
    """BM25Index 核心功能测试"""

    def test_add_and_search(self):
        """添加文档后能检索到匹配内容。"""
        index = BM25Index()
        chunks = [
            Chunk(
                content="Retrieval augmented generation",
                metadata=make_metadata(source_file="doc1.pdf"),
            ),
            Chunk(
                content="Machine learning transformers",
                metadata=make_metadata(source_file="doc2.pdf"),
            ),
        ]
        index.add_documents(chunks)

        results = index.search("retrieval", top_k=5)

        assert len(results) >= 1
        assert results[0].retrieval_method == "bm25"
        assert results[0].score > 0

    def test_add_and_search_chinese(self):
        """中文内容检索（基于字符级分词）。"""
        index = BM25Index()
        chunks = [
            Chunk(
                content="基于检索增强生成的问答系统",
                metadata=make_metadata(source_file="doc1.pdf"),
            ),
            Chunk(
                content="机器学习与深度学习模型对比",
                metadata=make_metadata(source_file="doc2.pdf"),
            ),
        ]
        index.add_documents(chunks)

        results = index.search("检索增强", top_k=5)

        assert len(results) >= 1
        assert results[0].retrieval_method == "bm25"

    def test_search_top_k(self):
        """top_k 参数限制返回数量。"""
        index = BM25Index()
        chunks = [
            Chunk(content=f"Document number {i}", metadata=make_metadata())
            for i in range(20)
        ]
        index.add_documents(chunks)

        results = index.search("document", top_k=5)
        assert len(results) <= 5

    def test_empty_index_returns_empty(self):
        """空索引检索返回空列表。"""
        index = BM25Index()
        assert index.search("anything", top_k=5) == []

    def test_clear_resets_index(self):
        """clear 后检索应返回空列表。"""
        index = BM25Index()
        index.add_documents(
            [
                Chunk(
                    content="something searchable",
                    metadata=make_metadata(),
                )
            ]
        )
        index.clear()
        assert index.search("something") == []

    def test_empty_query_returns_empty(self):
        """空查询字符串返回空列表。"""
        index = BM25Index()
        index.add_documents(
            [
                Chunk(
                    content="any content here",
                    metadata=make_metadata(),
                )
            ]
        )
        assert index.search("") == []
        assert index.search("   ") == []

    def test_save_and_load(self):
        """持久化后再加载应能检索。"""
        import os
        import tempfile

        index = BM25Index()
        chunks = [
            Chunk(
                content="Python is a great programming language",
                metadata=make_metadata(source_file="doc1.pdf"),
            ),
            Chunk(
                content="Java is also widely used",
                metadata=make_metadata(source_file="doc2.pdf"),
            ),
        ]
        index.add_documents(chunks)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            index.save(path)

            new_index = BM25Index()
            new_index.load(path)
            results = new_index.search("Python", top_k=5)

            assert len(results) >= 1
            assert results[0].retrieval_method == "bm25"
            assert "Python" in results[0].chunk.content
        finally:
            os.unlink(path)

    def test_create_bm25_index_factory(self):
        """工厂函数返回可用实例。"""
        from src.index import create_bm25_index

        index = create_bm25_index()
        assert index is not None
        assert isinstance(index, BM25Index)
