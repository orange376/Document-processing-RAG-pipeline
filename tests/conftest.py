"""Shared fixtures for the RAG pipeline test suite."""

import pytest
from src.domain.chunk import Chunk, ChunkMetadata, SearchResult


@pytest.fixture
def sample_chunk():
    """Return a minimal Chunk with metadata."""
    return Chunk(
        chunk_id="chk_test001",
        content="这是一个测试切片，包含示例内容。可用于检索验证。",
        metadata=ChunkMetadata(
            source_file="test_doc.docx",
            page_num=1,
            section="第一章",
            chunk_type="text",
            layout_tree_path=["第一章"],
            chunk_level="paragraph",
        ),
    )


@pytest.fixture
def sample_search_result(sample_chunk):
    """Return a SearchResult wrapping the sample chunk."""
    return SearchResult(chunk=sample_chunk, score=0.85, retrieval_method="hybrid")


@pytest.fixture
def multi_results():
    """Return 5 ranked search results with decreasing scores."""
    results = []
    for i in range(5):
        c = Chunk(
            content=f"文档内容段落{i+1}。这是第{i+1}段测试文本。包含足够的信息用于检索。",
            metadata=ChunkMetadata(
                source_file=f"doc_{i+1}.docx",
                page_num=i + 1,
                section=f"第{i+1}节",
                chunk_type="text",
                layout_tree_path=[f"第{i+1}节"],
            ),
        )
        results.append(SearchResult(chunk=c, score=0.95 - i * 0.1, retrieval_method="hybrid"))
    return results
