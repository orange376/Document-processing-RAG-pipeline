from __future__ import annotations

from dataclasses import dataclass, field
from .layout import BBox


@dataclass
class Cell:
    """表格单元格"""
    text: str
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


@dataclass
class Table:
    """还原后的表格"""
    bbox: BBox
    cells: list[Cell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    header_rows: int = 1
    is_page_break: bool = False
    caption: str = ""

    def to_markdown(self) -> str:
        """将表格渲染为 Markdown 格式"""
        if not self.cells or self.num_rows == 0:
            return ""

        # 构建二维矩阵
        matrix: list[list[str]] = [
            ["" for _ in range(self.num_cols)] for _ in range(self.num_rows)
        ]
        for cell in self.cells:
            r, c = cell.row_index, cell.col_index
            if 0 <= r < self.num_rows and 0 <= c < self.num_cols:
                matrix[r][c] = cell.text

        lines: list[str] = []
        for i, row in enumerate(matrix):
            lines.append("| " + " | ".join(row) + " |")
            if i == self.header_rows - 1:
                lines.append("|" + "|".join([" --- "] * self.num_cols) + "|")

        return "\n".join(lines)
