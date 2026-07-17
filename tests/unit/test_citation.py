from __future__ import annotations

from src.domain.chunk import CitationSource
from src.generation.citation import format_citation


class TestFormatCitation:
    def test_basic_format(self):
        """基本格式化：包含 source_file、page_num 和 section"""
        source = CitationSource(
            source_file="doc.pdf",
            page_num=3,
            section="Results",
            chunk_type="text",
            text="data",
        )
        result = format_citation(source)
        assert "doc.pdf" in result
        assert "第3页" in result
        assert "§Results" in result
        assert "📄" in result
        assert "📍" in result
        assert "📂" in result

    def test_missing_section(self):
        """section 为空时，省略 📂 § 部分"""
        source = CitationSource(
            source_file="report.pdf",
            page_num=5,
            section="",
            chunk_type="text",
            text="content",
        )
        result = format_citation(source)
        assert "report.pdf" in result
        assert "第5页" in result
        assert "📂" not in result
        assert "§" not in result

    def test_zero_page_num(self):
        """page_num 为 0 时仍正常显示"""
        source = CitationSource(
            source_file="appendix.pdf",
            page_num=0,
            section="Cover",
            chunk_type="text",
            text="title",
        )
        result = format_citation(source)
        assert "appendix.pdf" in result
        assert "第0页" in result
        assert "§Cover" in result

    def test_section_whitespace_only(self):
        """section 仅有空白字符时，简单实现按 truthy 处理，保留 📂 § 部分"""
        source = CitationSource(
            source_file="notes.pdf",
            page_num=2,
            section="   ",
            chunk_type="text",
            text="note",
        )
        result = format_citation(source)
        assert "notes.pdf" in result
        assert "第2页" in result
        assert "📂" in result
        assert "§   " in result
