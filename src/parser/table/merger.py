from __future__ import annotations

from src.domain import Table, Cell


class CrossPageTableMerger:
    """跨页表格拼接器"""

    def merge(self, tables: list[list[Table]]) -> list[Table]:
        """合并跨页断裂的表格

        Args:
            tables: 每页的表格列表（二维）

        Returns:
            合并后的表格列表
        """
        if not tables:
            return []

        flat = [t for page_tables in tables for t in page_tables]
        if len(flat) <= 1:
            return flat

        merged: list[Table] = []
        i = 0
        while i < len(flat):
            current = flat[i]
            # 检测下一页的第一个表格是否与当前表格连续
            if (
                i + 1 < len(flat)
                and current.num_cols > 0
                and flat[i + 1].num_cols == current.num_cols
                and flat[i + 1].header_rows == 0  # 下一页无表头 = 续表
            ):
                next_table = flat[i + 1]
                # 合并行
                for cell in next_table.cells:
                    new_cell = Cell(
                        text=cell.text,
                        row_index=cell.row_index + current.num_rows,
                        col_index=cell.col_index,
                        row_span=cell.row_span,
                        col_span=cell.col_span,
                        is_header=cell.is_header,
                    )
                    current.cells.append(new_cell)
                current.num_rows += next_table.num_rows
                current.is_page_break = True
                i += 1  # 跳过已合并的下一页

            merged.append(current)
            i += 1

        return merged
