import pytest
from src.domain import Document, Page, Block
from src.parser.chunker import StructureAwareChunker


@pytest.fixture
def sample_document():
    doc = Document(filename="test.pdf", file_type="pdf")
    page = Page(page_num=1, width=595, height=842)

    page.blocks = [
        Block(content="第一章 引言", block_type="title", page_num=1),
        Block(content="这是引言段落的内容...", block_type="text", page_num=1),
        Block(content="表格数据", block_type="table", page_num=1),
        Block(content="此后是一些正文。", block_type="text", page_num=1),
    ]
    doc.pages.append(page)

    page2 = Page(page_num=2, width=595, height=842)
    page2.blocks = [
        Block(content="1.1 背景介绍", block_type="section_heading", page_num=2),
        Block(content="背景内容段落。", block_type="text", page_num=2),
    ]
    doc.pages.append(page2)
    doc.total_pages = 2
    return doc


class TestStructureAwareChunker:
    def test_chunk_basic(self, sample_document):
        chunker = StructureAwareChunker(max_chunk_chars=2000)
        chunks = chunker.chunk(sample_document)

        # 预期切片：
        # 1. "第一章 引言" 之后的正文
        # 2. 表格（独立）
        # 3. 剩余正文
        # 4. "1.1 背景介绍" + 背景正文
        assert len(chunks) > 0

    def test_chunk_metadata(self, sample_document):
        chunker = StructureAwareChunker()
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            assert chunk.metadata is not None
            assert chunk.metadata.source_file == "test.pdf"
            assert chunk.metadata.page_num > 0
            assert chunk.metadata.chunk_type in ("text", "table", "formula", "figure")

    def test_chunk_empty_document(self):
        doc = Document(filename="empty.pdf", file_type="pdf")
        doc.pages.append(Page(page_num=1, width=595, height=842))
        chunker = StructureAwareChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

    def test_to_context_block_includes_metadata(self):
        chunker = StructureAwareChunker()
        doc = Document(filename="report.pdf", file_type="pdf")
        page = Page(page_num=3, width=595, height=842)
        page.blocks = [Block(content="重要数据", block_type="text", page_num=3)]
        doc.pages.append(page)
        doc.total_pages = 1

        chunks = chunker.chunk(doc)
        if chunks:
            block_text = chunks[0].to_context_block()
            assert "report.pdf" in block_text
            assert "第3页" in block_text
