from __future__ import annotations

from src.domain import Document, Chunk, ChunkMetadata, Page


class StructureAwareChunker:
    """结构感知切片器

    策略：
    1. 按标题层级（H1/H2/H3）作为切片边界
    2. 单一切片最大 token ≈ 512（可配置）
    3. 表/图/公式作为独立切片
    4. 每个切片携带完整元数据（源文件、页码、章节路径）
    """

    def __init__(self, max_chunk_chars: int = 1500):
        self.max_chunk_chars = max_chunk_chars

    def chunk(self, document: Document) -> list[Chunk]:
        """对文档执行结构感知切片"""
        chunks: list[Chunk] = []
        current_section = ""

        for page in document.pages:
            # 按 block 粒度切片
            page_chunks = self._chunk_page(page, document.filename, current_section)
            chunks.extend(page_chunks)

            # 更新当前章节（取 page 中最后一个标题）
            for block in page.blocks:
                if block.block_type in ("title", "heading", "section_heading"):
                    current_section = block.content[:60]

        return chunks

    def _chunk_page(
        self, page: Page, filename: str, default_section: str
    ) -> list[Chunk]:
        """对单页做切片"""
        chunks: list[Chunk] = []
        buffer = ""
        buffer_start_bbox = None
        section = default_section

        for block in page.blocks:
            # 标题作为切片边界
            if block.block_type in ("title", "heading", "section_heading"):
                # 提交当前 buffer
                if buffer.strip():
                    chunks.append(self._make_chunk(
                        buffer.strip(), filename, page.page_num,
                        section, block.block_type, buffer_start_bbox,
                    ))
                    buffer = ""
                    buffer_start_bbox = None
                section = block.content[:60]
                continue

            # 表格/公式/图片作为独立切片
            if block.block_type in ("table", "formula", "figure"):
                if buffer.strip():
                    chunks.append(self._make_chunk(
                        buffer.strip(), filename, page.page_num,
                        section, "text", buffer_start_bbox,
                    ))
                    buffer = ""
                    buffer_start_bbox = None
                chunks.append(self._make_chunk(
                    block.content, filename, page.page_num,
                    section, block.block_type, block.bbox,
                ))
                continue

            # 普通文本：累积到 buffer
            if not buffer:
                buffer_start_bbox = block.bbox
            buffer += block.content + "\n"

            # 超长截断
            if len(buffer) >= self.max_chunk_chars:
                chunks.append(self._make_chunk(
                    buffer.strip(), filename, page.page_num,
                    section, "text", buffer_start_bbox,
                ))
                buffer = ""
                buffer_start_bbox = None

        # 提交剩余 buffer
        if buffer.strip():
            chunks.append(self._make_chunk(
                buffer.strip(), filename, page.page_num,
                section, "text", buffer_start_bbox,
            ))

        return chunks

    def _make_chunk(
        self, content: str, filename: str, page_num: int,
        section: str, chunk_type: str, bbox=None,
    ) -> Chunk:
        return Chunk(
            content=content,
            metadata=ChunkMetadata(
                source_file=filename,
                page_num=page_num,
                section=section,
                chunk_type=chunk_type,
                bbox=bbox,
            ),
        )
