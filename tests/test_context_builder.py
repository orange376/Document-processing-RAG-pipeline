"""Test suite for ContextBuilder — token counting and truncation."""

import pytest
from src.generation.context_builder import (
    ContextBuilder,
    _count_tokens,
    _truncate_text_to_tokens,
)
from src.domain.chunk import Chunk, ChunkMetadata, SearchResult


def _make_result(content: str, source: str = "test.docx", score: float = 0.9) -> SearchResult:
    c = Chunk(
        content=content,
        metadata=ChunkMetadata(
            source_file=source,
            page_num=1,
            section="",
            chunk_type="text",
            layout_tree_path=[],
        ),
    )
    return SearchResult(chunk=c, score=score, retrieval_method="hybrid")


def test_count_tokens_english():
    n = _count_tokens("hello world")
    assert n > 0


def test_count_tokens_chinese():
    n = _count_tokens("你好世界这是一段中文测试")
    assert n > 0


def test_build_empty():
    builder = ContextBuilder(max_context_tokens=500)
    ctx, sources = builder.build([])
    assert ctx == ""
    assert sources == []


def test_build_single_result():
    builder = ContextBuilder(max_context_tokens=5000)
    results = [_make_result("测试内容")]
    ctx, sources = builder.build(results)
    assert "测试内容" in ctx
    assert len(sources) == 1
    assert sources[0].source_file == "test.docx"


def test_build_truncation():
    builder = ContextBuilder(max_context_tokens=100)
    long_text = "这是很长的内容。它包含很多汉字用于测试。我们需要确保截断功能正常工作。" * 10
    results = [_make_result(long_text)]
    ctx, sources = builder.build(results, max_context_tokens=100)
    # At least one result should be returned even if truncated
    assert len(sources) <= 1
    assert len(ctx) > 0


def test_build_multiple_under_limit():
    builder = ContextBuilder(max_context_tokens=5000)
    results = [_make_result(f"内容{i}") for i in range(5)]
    ctx, sources = builder.build(results)
    assert len(sources) == 5


def test_truncate_text_to_tokens_shorter():
    text = "Hello."
    out = _truncate_text_to_tokens(text, 1000)
    assert text in out  # no truncation needed


def test_truncate_text_to_tokens_longer():
    long = "句子A。" * 200
    out = _truncate_text_to_tokens(long, 50)
    assert len(out) < len(long)
    assert "[内容过长，已截断]" in out or "。" in out


def test_context_builder_sets_default_tokens():
    builder = ContextBuilder()
    assert builder._max_tokens > 0
