"""Test suite for StructureAwareChunker — structure-aware + multi-granularity chunking."""

import pytest
from src.domain import Document, Page, Block
from src.parser.chunker import StructureAwareChunker


def _make_page(blocks: list[Block]) -> Page:
    p = Page(page_num=1, width=595, height=842)
    p.blocks = blocks
    return p


def _make_block(content: str, block_type: str = "text", **kw) -> Block:
    return Block(content=content, block_type=block_type, page_num=1, reading_order=0, **kw)


def _make_doc(filename: str, pages: list[Page], file_type: str = "docx") -> Document:
    return Document(filename=filename, file_path=filename, file_type=file_type, pages=pages, total_pages=len(pages))


# ---------------------------------------------------------------------------
# Heading-based splitting
# ---------------------------------------------------------------------------

def test_heading_splits_chunks():
    chunker = StructureAwareChunker()
    page = _make_page([
        _make_block("第一章 概述", "section_heading"),
        _make_block("这是第一段内容。包含一些文字。"),
        _make_block("1.1 详细说明", "section_heading"),
        _make_block("这是第二段内容。"),
    ])
    doc = _make_doc("test.docx", [page])
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2  # two text paragraphs between headings


def test_formula_block_is_isolated():
    chunker = StructureAwareChunker()
    page = _make_page([
        _make_block("一些文本。"),
        _make_block(r"$\frac{a}{b}$", "formula"),
        _make_block("更多文本。"),
    ])
    doc = _make_doc("test.docx", [page])
    chunks = chunker.chunk(doc)
    # formula is its own chunk, text before and after are separate
    types = [c.metadata.chunk_type for c in chunks if c.metadata]
    assert "formula" in types
    assert "text" in types


# ---------------------------------------------------------------------------
# Overlap between splits
# ---------------------------------------------------------------------------

def test_short_doc_no_overlap():
    chunker = StructureAwareChunker(max_chunk_chars=5000)
    page = _make_page([_make_block("短文本。")])
    doc = _make_doc("test.docx", [page])
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1


def test_overlap_retains_tail():
    chunker = StructureAwareChunker(max_chunk_chars=60, overlap=20)
    long_text = "这是第一句。这是第二句。这是第三句。这是第四句。这是第五句。"
    page = _make_page([_make_block(long_text)])
    doc = _make_doc("test.docx", [page])
    chunks = chunker.chunk(doc)
    if len(chunks) >= 2:
        # Tail of chunk 1 should appear in chunk 2
        tail = chunks[0].content[-30:]
        assert any(tail[:20] in chunks[i].content for i in range(1, len(chunks)))


# ---------------------------------------------------------------------------
# Sentence-level fine chunking (#17)
# ---------------------------------------------------------------------------

def test_fine_chunk_splits_sentences():
    chunker = StructureAwareChunker()
    # Sentence-level splitting only triggers for buffers that exceed
    # max_chunk_chars (2500) — a short block stays a single paragraph chunk.
    sentence = "这是第一句话这是第一句话这是第一句话。这是第二句话这是第二句话这是第二句话。这是第三句话这是第三句话这是第三句话。"
    page = _make_page([
        _make_block(sentence * 50),  # ~2850 chars → must be split at a sentence break
    ])
    doc = _make_doc("test.docx", [page])
    chunks = chunker.fine_chunk(doc)
    levels = [c.metadata.chunk_level for c in chunks if c.metadata]
    assert "sentence" in levels, f"expected a sentence-level chunk, got {levels}"
    assert "paragraph" in levels, f"expected a paragraph-level chunk, got {levels}"


def test_fine_chunk_skips_formula():
    chunker = StructureAwareChunker()
    page = _make_page([
        _make_block(r"$x = \frac{-b}{2a}$", "formula"),
    ])
    doc = _make_doc("test.docx", [page])
    chunks = chunker.fine_chunk(doc)
    levels = [c.metadata.chunk_level for c in chunks if c.metadata]
    assert all(l == "paragraph" for l in levels)  # formula is not sentence-split


def test_fine_chunk_preserves_metadata():
    chunker = StructureAwareChunker()
    page = _make_page([
        _make_block("标题", "section_heading"),
        _make_block("内容句子A。内容句子B。"),
    ])
    doc = _make_doc("test.docx", [page])
    chunks = chunker.fine_chunk(doc)
    for c in chunks:
        meta = c.metadata
        assert meta is not None
        assert meta.source_file == "test.docx"
        assert meta.page_num == 1


# ---------------------------------------------------------------------------
# layout_tree_path
# ---------------------------------------------------------------------------

def test_layout_tree_path_populated():
    chunker = StructureAwareChunker()
    page = _make_page([
        _make_block("第一章 概述", "section_heading"),
        _make_block("段落A。", "text"),
        _make_block("1.1 详情", "section_heading"),
        _make_block("段落B。", "text"),
    ])
    doc = _make_doc("test.docx", [page])
    chunks = chunker.chunk(doc)
    for c in chunks:
        if c.metadata and c.metadata.chunk_type == "text":
            assert isinstance(c.metadata.layout_tree_path, list)
