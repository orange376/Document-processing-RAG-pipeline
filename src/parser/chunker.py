from __future__ import annotations

import re

from src.domain import Document, Chunk, ChunkMetadata, Page


# Match lines that start with a question number — these are split points
# within a text section (e.g. "1．" "(1)" "（2）" "①").
# Does NOT match option labels (A． B． C． D．) — those stay with the question.
_QUESTION_NUM_RE = re.compile(
    r'^(\d+[．\.、]'
    r'|[（(]\d+[）)]'
    r'|[①②③④⑤⑥⑦⑧⑨⑩]'
    r'|[一二三四五六七八九十]+[、．])'
)

_SENTENCE_BOUNDARY_RE = re.compile(r'[。！？\n]')


class StructureAwareChunker:
    """结构感知切片器

    策略：
    1. 按标题层级（H1/H2/H3）作为切片边界
    2. 单一切片最大 token ≈ 512（可配置）
    3. 表/图/公式作为独立切片
    4. 每个切片携带完整元数据（源文件、页码、章节路径、版面树路径）
    5. 相邻切片保留重叠区域（默认 128 字符），防止边界语义断裂
    """

    def __init__(self, max_chunk_chars: int = 1500, overlap: int = 128):
        self.max_chunk_chars = max_chunk_chars
        self.overlap = overlap

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
        heading_stack: list[str] = []  # heading hierarchy → layout_tree_path

        for block in page.blocks:
            # 标题作为切片边界
            if block.block_type in ("title", "heading", "section_heading"):
                # 提交当前 buffer（使用 heading_stack 快照）
                if buffer.strip():
                    chunks.append(self._make_chunk(
                        buffer.strip(), filename, page.page_num,
                        section, block.block_type, buffer_start_bbox,
                        layout_tree_path=list(heading_stack),
                    ))
                    buffer = ""
                    buffer_start_bbox = None
                section = block.content[:60]
                # 更新标题栈：追加新标题（保留父级层级信息）
                heading_stack.append(section)
                continue

            # 表格/公式/图片作为独立切片
            if block.block_type in ("table", "formula", "figure"):
                if buffer.strip():
                    chunks.append(self._make_chunk(
                        buffer.strip(), filename, page.page_num,
                        section, "text", buffer_start_bbox,
                        layout_tree_path=list(heading_stack),
                    ))
                    buffer = ""
                    buffer_start_bbox = None
                chunks.append(self._make_chunk(
                    block.content, filename, page.page_num,
                    section, block.block_type, block.bbox,
                    layout_tree_path=list(heading_stack),
                ))
                continue

            # 普通文本：累积到 buffer
            # 若 block 以题号开头（如 "1．" "（2）"），先提交当前 buffer 再开始新累积
            if buffer and _QUESTION_NUM_RE.match(block.content.strip()):
                chunks.append(self._make_chunk(
                    buffer.strip(), filename, page.page_num,
                    section, "text", buffer_start_bbox,
                    layout_tree_path=list(heading_stack),
                ))
                buffer = ""
                buffer_start_bbox = None

            if not buffer:
                buffer_start_bbox = block.bbox
            buffer += block.content + "\n"

            # 超长截断 + 重叠（#13: 保留尾部 overlap 字符到下一切片）
            if len(buffer) >= self.max_chunk_chars:
                split = self._find_split_point(buffer, self.max_chunk_chars)
                chunks.append(self._make_chunk(
                    buffer[:split].strip(), filename, page.page_num,
                    section, "text", buffer_start_bbox,
                    layout_tree_path=list(heading_stack),
                ))
                # 保留尾部 overlap 字符，防止边界语义断裂
                overlap_start = max(0, split - self.overlap)
                buffer = buffer[overlap_start:]
                buffer_start_bbox = None

        # 提交剩余 buffer
        if buffer.strip():
            chunks.append(self._make_chunk(
                buffer.strip(), filename, page.page_num,
                section, "text", buffer_start_bbox,
                layout_tree_path=list(heading_stack),
            ))

        return chunks

    def _find_split_point(self, text: str, max_len: int) -> int:
        """在 max_len 附近找到合适的切分点（优先句末对齐）。"""
        if len(text) <= max_len:
            return len(text)
        # 向后查找到句末边界
        candidate = max_len
        for boundary in ['。', '！', '？', '\n']:
            pos = text.rfind(boundary, 0, max_len)
            if pos > max_len // 2:  # 只在合理位置才使用句末对齐
                candidate = max(candidate, pos + 1)
                break
        return min(candidate, max_len)

    def _make_chunk(
        self, content: str, filename: str, page_num: int,
        section: str, chunk_type: str, bbox=None,
        layout_tree_path: list[str] | None = None,
    ) -> Chunk:
        return Chunk(
            content=content,
            metadata=ChunkMetadata(
                source_file=filename,
                page_num=page_num,
                section=section,
                chunk_type=chunk_type,
                bbox=bbox,
                layout_tree_path=layout_tree_path or [],
            ),
        )
