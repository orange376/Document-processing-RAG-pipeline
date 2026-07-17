import pytest
from src.domain import Table, Cell, LayoutElement, BBox
from src.parser.table import TableDetector, CrossPageTableMerger


class TestTableDetector:
    def test_detect(self):
        elements = [
            LayoutElement(bbox=BBox(0, 0, 100, 20, 1), category="text",
                          confidence=0.9, text="段落", reading_order=0),
            LayoutElement(bbox=BBox(0, 25, 100, 80, 1), category="table",
                          confidence=0.9, text="表格数据", reading_order=1),
        ]
        detector = TableDetector()
        result = detector.detect(elements)
        assert len(result) == 1
        assert result[0].category == "table"


class TestCrossPageTableMerger:
    def test_merge_no_tables(self):
        merger = CrossPageTableMerger()
        assert merger.merge([]) == []
        assert merger.merge([[], []]) == []

    def test_merge_single_page(self):
        merger = CrossPageTableMerger()
        table = Table(num_rows=3, num_cols=2, bbox=None)  # type: ignore
        result = merger.merge([[table]])
        assert len(result) == 1

    def test_merge_cross_page(self):
        merger = CrossPageTableMerger()
        t1 = Table(num_rows=10, num_cols=3, header_rows=1, bbox=None)  # type: ignore
        for i in range(10):
            t1.cells.append(Cell(text=f"R{i}", row_index=i, col_index=0))

        t2 = Table(num_rows=5, num_cols=3, header_rows=0, bbox=None)  # type: ignore
        for i in range(5):
            t2.cells.append(Cell(text=f"R{i+10}", row_index=i, col_index=0))

        result = merger.merge([[t1], [t2]])
        assert len(result) == 1  # 合并成功
        assert result[0].num_rows == 15
        assert result[0].is_page_break is True
