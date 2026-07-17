from __future__ import annotations

from src.domain import LayoutElement, Table, BBox, Cell


class TableDetector:
    """从版面元素中筛选出表格区域"""

    TABLE_CATEGORIES = {"table", "table_caption", "table_content"}

    def detect(self, elements: list[LayoutElement]) -> list[LayoutElement]:
        return [el for el in elements if el.category.lower() in self.TABLE_CATEGORIES]

    def detect_as_table_objects(
        self, elements: list[LayoutElement], page_num: int = 0
    ) -> list[Table]:
        """将表格 LayoutElement 转为 Table 对象（占位，等待结构还原）"""
        tables = []
        for el in self.detect(elements):
            if el.category.lower() == "table":
                tables.append(Table(
                    bbox=BBox(
                        x0=el.bbox.x0, y0=el.bbox.y0,
                        x1=el.bbox.x1, y1=el.bbox.y1,
                        page_num=page_num,
                    ),
                ))
        return tables
