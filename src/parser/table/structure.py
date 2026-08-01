from __future__ import annotations

import logging

from src.domain import Table, Cell

logger = logging.getLogger(__name__)


class TableStructureRecoverer:
    """表格结构还原器

    策略：
    1. 从 easyocr 文本 + 坐标推测行列结构（轻量）
    2. 从表格截图调用 Qwen-VL 提取 Markdown 表格（精确，推荐）
    """

    def recover(self, table: Table, ocr_text: str | None = None) -> Table:
        """基础版：从 OCR 文本粗粒度还原"""
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

    async def recover_from_image(
        self, img_bytes: bytes, table: Table | None = None
    ) -> Table | None:
        """使用 Qwen-VL 从表格截图提取 Markdown 并解析为 Table 对象。

        Args:
            img_bytes: 表格截图（PNG/JPEG 字节）。
            table: 可选的已有 Table 对象（用于填充 bbox 信息）。

        Returns:
            解析后的 Table，或 None（无法识别时）。
        """
        markdown = await self._qwen_vl_table_to_markdown(img_bytes)
        if not markdown:
            return None
        return self._parse_markdown_table(markdown, table)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _qwen_vl_table_to_markdown(self, img_bytes: bytes) -> str | None:
        """Call Qwen-VL to extract a Markdown table from an image."""
        try:
            from src.generation.llm_client import LLMClient

            llm = LLMClient(provider="qwen")
            prompt = (
                "请将此图片中的表格转换为 Markdown 表格格式。\n"
                "要求：\n"
                "1. 保留表头行\n"
                "2. 保留所有单元格内容，不要遗漏\n"
                "3. 合并的单元格展开为独立单元格\n"
                "4. 只输出 Markdown 表格，不要输出其他解释文字\n"
                "5. 如果图片中不包含表格，仅回复 NOT_A_TABLE"
            )
            result = await llm.chat_with_image(
                prompt,
                img_bytes,
                system="你是一个表格结构识别专家。精确地将表格图片转换为 Markdown 表格。",
            )
            if result and "NOT_A_TABLE" not in result.upper():
                return result.strip()
        except Exception:
            logger.exception("Qwen-VL table recognition failed")
        return None

    def _parse_markdown_table(
        self, markdown: str, table: Table | None = None
    ) -> Table | None:
        """Parse a Markdown table string into a Table domain object.

        Example input::

            | Name | Age | City |
            | ---- | --- | ---- |
            | Tom  | 30  | NY   |
        """
        lines = [l.strip() for l in markdown.split("\n") if l.strip()]
        if len(lines) < 2:
            return None

        # Separate header row and data rows (skip separator line)
        header_line = lines[0]
        data_start = 2 if len(lines) >= 2 and set(lines[1].strip()) <= {"|", "-", " "} else 1
        data_lines = lines[data_start:]

        # Parse columns from header
        headers = [h.strip() for h in header_line.strip("|").split("|")]
        num_cols = len(headers)

        cells: list[Cell] = []

        # Header cells
        for ci, h in enumerate(headers):
            cells.append(Cell(text=h, row_index=0, col_index=ci, is_header=True))

        # Data cells
        for ri, row_str in enumerate(data_lines):
            values = [v.strip() for v in row_str.strip("|").split("|")]
            for ci, val in enumerate(values):
                if ci < num_cols:
                    cells.append(Cell(
                        text=val,
                        row_index=ri + 1,
                        col_index=ci,
                        is_header=False,
                    ))

        table = table or Table(
            bbox=table.bbox if table else None,
            cells=cells,
            num_rows=len(data_lines) + 1,
            num_cols=num_cols,
            header_rows=1,
        )
        table.cells = cells
        table.num_rows = len(data_lines) + 1
        table.num_cols = num_cols
        return table
