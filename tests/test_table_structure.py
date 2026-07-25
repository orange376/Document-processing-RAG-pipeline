"""Test suite for TableStructureRecoverer — markdown table parsing."""

import pytest
from src.parser.table.structure import TableStructureRecoverer


def test_parse_simple_markdown_table():
    r = TableStructureRecoverer()
    md = "| Name | Age | City |\n|------|-----|------|\n| Tom  | 30  | NY   |\n| Bob  | 25  | LA   |"
    table = r._parse_markdown_table(md)
    assert table is not None
    assert table.num_rows == 3  # header + 2 data
    assert table.num_cols == 3
    assert len(table.cells) == 9  # 3×3
    # Header cells are marked
    headers = [c for c in table.cells if c.is_header]
    assert len(headers) == 3
    assert headers[0].text == "Name"


def test_parse_single_row_table():
    r = TableStructureRecoverer()
    md = "| A | B |\n|-----|----|"
    table = r._parse_markdown_table(md)
    assert table is not None
    assert table.num_rows == 1
    assert table.num_cols == 2
    assert all(c.is_header for c in table.cells)


def test_parse_empty_returns_none():
    r = TableStructureRecoverer()
    assert r._parse_markdown_table("") is None
    assert r._parse_markdown_table("not a table") is None


def test_parse_with_extra_pipes():
    r = TableStructureRecoverer()
    md = "|  ID  |  Value  |\n|  --- |  ---   |\n|  1   |   100   |"
    table = r._parse_markdown_table(md)
    assert table is not None
    assert table.num_cols == 2
    assert table.cells[0].text == "ID"


def test_parse_multi_row():
    r = TableStructureRecoverer()
    md = "| Col1 | Col2 |\n|------|------|\n| a1 | b1 |\n| a2 | b2 |\n| a3 | b3 |"
    table = r._parse_markdown_table(md)
    assert table is not None
    assert table.num_rows == 4
    assert table.num_cols == 2
    # Last data cell
    non_header = [c for c in table.cells if not c.is_header]
    assert non_header[-1].text == "b3"


def test_recover_from_text_fallback():
    r = TableStructureRecoverer()
    from src.domain import Table, BBox, Cell
    t = Table(bbox=BBox(0, 0, 100, 100, 1))
    result = r.recover(t, ocr_text="line1\nline2\nline3")
    assert result.num_rows == 3
    assert result.num_cols == 1
