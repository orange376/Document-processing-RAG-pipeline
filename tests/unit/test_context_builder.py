import pytest

from src.domain.chunk import Chunk, ChunkMetadata, SearchResult
from src.generation.context_builder import ContextBuilder


class TestContextBuilder:
    def test_build_empty(self):
        """空输入应返回空字符串和空列表。"""
        cb = ContextBuilder()
        ctx, sources = cb.build([])
        assert ctx == ""
        assert sources == []

    def test_build_single_result(self):
        """单个结果应生成带编号的上下文块和对应的引用源。"""
        cb = ContextBuilder()
        chunk = Chunk(
            content="RAG 是一种有效的检索增强生成技术。",
            metadata=ChunkMetadata(
                source_file="paper.pdf",
                page_num=3,
                section="2.1",
                chunk_type="text",
            ),
        )
        results = [SearchResult(chunk=chunk, score=0.95)]
        ctx, sources = cb.build(results)

        assert "[1]" in ctx
        assert "RAG 是一种有效的检索增强生成技术。" in ctx
        assert "paper.pdf" in ctx
        assert "第3页" in ctx
        assert len(sources) == 1
        assert sources[0].source_file == "paper.pdf"
        assert sources[0].page_num == 3
        assert sources[0].section == "2.1"
        assert sources[0].chunk_type == "text"
        assert sources[0].text == "RAG 是一种有效的检索增强生成技术。"

    def test_build_multiple_results(self):
        """多个结果应依次编号，并用双换行分隔。"""
        cb = ContextBuilder()
        chunk1 = Chunk(
            content="内容一",
            metadata=ChunkMetadata(
                source_file="a.pdf", page_num=1, section="1", chunk_type="text",
            ),
        )
        chunk2 = Chunk(
            content="内容二",
            metadata=ChunkMetadata(
                source_file="b.pdf", page_num=2, section="2", chunk_type="table",
            ),
        )
        results = [
            SearchResult(chunk=chunk1, score=0.9),
            SearchResult(chunk=chunk2, score=0.8),
        ]
        ctx, sources = cb.build(results)

        assert ctx.startswith("[1]")
        assert "[2]" in ctx
        assert ctx.count("\n\n") == 1  # 两个块之间有一个双换行
        assert len(sources) == 2

    def test_build_result_without_metadata(self):
        """结果缺少 metadata 时不应生成 CitationSource。"""
        cb = ContextBuilder()
        chunk = Chunk(content="无元数据内容")
        results = [SearchResult(chunk=chunk, score=0.5)]
        ctx, sources = cb.build(results)

        assert "[1]" in ctx
        assert "无元数据内容" in ctx
        assert sources == []  # 没有 metadata 就没有 CitationSource

    def test_build_mixed_metadata(self):
        """部分结果有 metadata，部分没有。"""
        cb = ContextBuilder()
        chunk_with_meta = Chunk(
            content="有元数据",
            metadata=ChunkMetadata(
                source_file="x.pdf", page_num=1, section="1", chunk_type="text",
            ),
        )
        chunk_without_meta = Chunk(content="无元数据")
        results = [
            SearchResult(chunk=chunk_with_meta, score=0.9),
            SearchResult(chunk=chunk_without_meta, score=0.5),
        ]
        ctx, sources = cb.build(results)

        assert len(sources) == 1
        assert sources[0].source_file == "x.pdf"
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_build_join_separator(self):
        """验证多个块之间使用双换行分隔。"""
        cb = ContextBuilder()
        chunks = [
            Chunk(
                content=f"内容{i}",
                metadata=ChunkMetadata(
                    source_file=f"{i}.pdf", page_num=i, section=str(i), chunk_type="text",
                ),
            )
            for i in range(1, 4)
        ]
        results = [SearchResult(chunk=c, score=0.9) for c in chunks]
        ctx, sources = cb.build(results)

        expected_separator = "\n\n"
        assert ctx.count(expected_separator) == 2  # 3 个块之间有 2 个分隔符
        assert len(sources) == 3
