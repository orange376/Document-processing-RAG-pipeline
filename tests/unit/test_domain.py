import pytest
from src.domain import BBox, Table, Cell, Chunk, ChunkMetadata, CitationSource, QueryResult, SearchResult


class TestBBox:
    def test_properties(self):
        bbox = BBox(0, 0, 100, 200, page_num=1)
        assert bbox.width == 100
        assert bbox.height == 200
        assert bbox.area == 20000

    def test_iou_no_overlap(self):
        a = BBox(0, 0, 10, 10, page_num=1)
        b = BBox(20, 20, 30, 30, page_num=1)
        assert a.iou(b) == 0.0

    def test_iou_perfect(self):
        a = BBox(0, 0, 10, 10, page_num=1)
        b = BBox(0, 0, 10, 10, page_num=1)
        assert a.iou(b) == 1.0

    def test_iou_partial(self):
        a = BBox(0, 0, 10, 10, page_num=1)
        b = BBox(5, 0, 15, 10, page_num=1)
        assert 0.3 < a.iou(b) < 0.35


class TestTable:
    def test_to_markdown_simple(self):
        table = Table(
            bbox=None,  # type: ignore
            num_rows=3, num_cols=2, header_rows=1,
            cells=[
                Cell("Name", 0, 0, is_header=True),
                Cell("Age", 0, 1, is_header=True),
                Cell("Alice", 1, 0),
                Cell("30", 1, 1),
                Cell("Bob", 2, 0),
                Cell("25", 2, 1),
            ],
        )
        md = table.to_markdown()
        assert "| Name | Age |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 30 |" in md


class TestChunk:
    def test_to_context_block(self):
        chunk = Chunk(
            content="这是一个测试段落。",
            metadata=ChunkMetadata(
                source_file="test.pdf",
                page_num=3,
                section="2.1",
                chunk_type="text",
            ),
        )
        block = chunk.to_context_block()
        assert "test.pdf" in block
        assert "第3页" in block
        assert "§2.1" in block
        assert "这是一个测试段落。" in block


class TestSearchResult:
    def test_default_retrieval_method(self):
        chunk = Chunk(content="test")
        result = SearchResult(chunk=chunk, score=0.95)
        assert result.score == 0.95
        assert result.retrieval_method == "hybrid"
        assert result.chunk is chunk


class TestCitationSource:
    def test_field_access(self):
        source = CitationSource(
            source_file="report.pdf",
            page_num=5,
            section="3.2",
            chunk_type="text",
            text="Some cited content.",
        )
        assert source.source_file == "report.pdf"
        assert source.page_num == 5
        assert source.section == "3.2"
        assert source.chunk_type == "text"
        assert source.text == "Some cited content."


class TestQueryResult:
    def test_creation_with_citations(self):
        citations = [
            CitationSource(
                source_file="a.pdf", page_num=1,
                section="1", chunk_type="text", text="A",
            ),
            CitationSource(
                source_file="b.pdf", page_num=2,
                section="2", chunk_type="table", text="B",
            ),
        ]
        result = QueryResult(
            answer="The answer is 42.",
            citations=citations,
        )
        assert result.answer == "The answer is 42."
        assert len(result.citations) == 2
        assert result.citations[0].source_file == "a.pdf"

    def test_confidence_details_default(self):
        citations = [
            CitationSource(
                source_file="x.pdf", page_num=1,
                section="1", chunk_type="text", text="X",
            ),
        ]
        result = QueryResult(answer="Test", citations=citations)
        assert result.confidence_details == {}
        assert result.needs_review is False
        assert result.confidence == 0.0
