from __future__ import annotations

from src.domain import Table, Cell


class TableStructureRecoverer:
    """表格结构还原器

    策略：
    1. 从 OCR 输出的文本 + 坐标推测行列结构
    2. 按 y 坐标聚类得到行，按 x 坐标聚类得到列
    3. 合并单元格通过 overlap 检测
    """

    def recover(self, table: Table, ocr_text: str | None = None) -> Table:
        """基础版：从 OCR 文本粗粒度还原

        复杂场景留给后续 PaddleOCR-VL-1.5 内置表格识别
        """
        if ocr_text:
            lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
            if lines and table.num_rows == 0:
                table.num_rows = len(lines)
                table.num_cols = 1
                for i, line in enumerate(lines):
                    table.cells.append(
                        Cell(text=line, row_index=i, col_index=0)
                    )
        return table
