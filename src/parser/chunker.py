from __future__ import annotations

import re

from src.domain import Document, Chunk, ChunkMetadata, Page


# Match lines that start with a question number — these are split points
# within a text section (e.g. "1．" "(1)" "（2）" "①").
_QUESTION_NUM_RE = re.compile(
    r"^(\d+[．\.、]\s*"
    r"|[（(]\d+[）)]"
    r"|[①②③④⑤⑥⑦⑧⑨⑩]"
    r"|[一二三四五六七八九十]+[、．])"
)

# Option labels that should stay attached to the preceding question chunk.
# e.g. "A．1；    B．2；    C．3；    D．4"
_OPTION_RE = re.compile(r"^[A-Da-d][．\.、]")

# Block is purely options (starts with option letter) — merge into preceding chunk
_OPTIONS_ONLY_RE = re.compile(
    r"^(?:[A-Da-d][．\.、]|\[IMG_\d+\][；;]\s*[B-Db-d][．\.、])"
)

# Split point for multiple questions in one block: "8.xxx  9．xxx"
_INLINE_QUESTION_SPLIT = re.compile(r"\s{2,}(?=\d+[．\.、])")

# Standalone short headings that are better merged into the next chunk
_SHORT_HEADING_NAMES = {"填空题", "判断题", "选择题", "解答题", "计算题", "证明题"}

_DEFAULT_MAX_CHARS: int = 2500


class StructureAwareChunker:
    """结构感知切片器

    策略：
    1. 按标题层级作为切片边界
    2. 一道题 = 一个切片（题号开头的 block 起新片）
    3. 表/图/公式作为独立切片
    4. 选择题选项黏合到题干，不独立成片
    5. 短标题（"填空题""判断题"）合并到后续内容
    6. 单一切片最大 ~2500 字符，超出按段落边界截断
    7. 相邻切片保留重叠区域（128 字符），防边界语义断裂
    8. 每个切片携带完整元数据（源文件、页码、章节路径）
    """

    def __init__(self, max_chunk_chars: int = _DEFAULT_MAX_CHARS, overlap: int = 128):
        self.max_chunk_chars = max_chunk_chars
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, document: Document) -> list[Chunk]:
        """对文档执行结构感知切片。"""
        chunks: list[Chunk] = []
        current_section = ""

        for page in document.pages:
            page_chunks = self._chunk_page(page, document.filename, current_section)
            chunks.extend(page_chunks)
            for block in page.blocks:
                if block.block_type in ("title", "heading", "section_heading"):
                    current_section = block.content[:60]

        # Post-processing: merge orphans
        chunks = self._merge_option_orphans(chunks)
        chunks = self._merge_stub_headings(chunks)
        return chunks

    def fine_chunk(self, document: Document) -> list[Chunk]:
        """Multi-granularity chunking — returns paragraph-level chunks only.

        Sentence-level splitting is intentionally NOT performed:
        paragraphs are already the right granularity for exam questions
        and narrative text.  Splitting at sentences produces chunks
        that are too small to carry complete question context.
        """
        return self.chunk(document)

    # ------------------------------------------------------------------
    # Per-page
    # ------------------------------------------------------------------

    def _chunk_page(
        self, page: Page, filename: str, default_section: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer = ""
        buffer_start_bbox = None
        section = default_section
        heading_stack: list[str] = []

        for block in page.blocks:
            # ── Heading blocks ──
            if block.block_type in ("title", "heading", "section_heading"):
                h_text = block.content.strip()
                # Short generic headings ("填空题", "判断题"): start new section
                # Flush current buffer first (heading belongs to NEXT content)
                if h_text in _SHORT_HEADING_NAMES:
                    if buffer.strip():
                        chunks.append(self._make_chunk(
                            buffer.strip(), filename, page.page_num,
                            section, "text", buffer_start_bbox,
                            layout_tree_path=list(heading_stack),
                        ))
                        buffer = ""
                        buffer_start_bbox = None
                    buffer = h_text + "\n"
                    buffer_start_bbox = block.bbox
                    section = h_text
                    heading_stack.append(h_text)
                    continue
                # Real heading — flush buffer, then start new buffer WITH heading text
                # (the heading IS often the question stem, e.g. "2. 设a与b平行...")
                if buffer.strip():
                    chunks.append(self._make_chunk(
                        buffer.strip(), filename, page.page_num,
                        section, "text", buffer_start_bbox,
                        layout_tree_path=list(heading_stack),
                    ))
                section = block.content[:60]
                heading_stack.append(section)
                # Start new buffer with this heading's content
                buffer = h_text + "\n"
                buffer_start_bbox = block.bbox
                continue

            # ── Table / formula / figure → standalone chunks ──
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

            # ── Text block ──
            text = block.content.strip()
            if not text:
                continue

            # Split block if it contains multiple questions inline
            # e.g. "8.xxx            9．xxx"
            sub_blocks = _INLINE_QUESTION_SPLIT.split(text) if len(text) > 40 else [text]

            for si, sub_text in enumerate(sub_blocks):
                sub_text = sub_text.strip()
                if not sub_text:
                    continue

                # Question-number boundary: flush and start fresh
                # (skip for the first sub-block of an inline split —
                #  it may merge with previous buffer content)
                is_new_question = (
                    bool(_QUESTION_NUM_RE.match(sub_text))
                    and not _OPTIONS_ONLY_RE.match(sub_text)
                )
                if is_new_question and buffer.strip():
                    # Don't flush if buffer is just a short heading prefix
                    buf_stripped = buffer.strip()
                    if buf_stripped not in _SHORT_HEADING_NAMES:
                        chunks.append(self._make_chunk(
                            buf_stripped, filename, page.page_num,
                            section, "text", buffer_start_bbox,
                            layout_tree_path=list(heading_stack),
                        ))
                        buffer = ""
                        buffer_start_bbox = None

                if not buffer:
                    buffer_start_bbox = block.bbox
                buffer += sub_text + "\n"

                # Oversized buffer → split at a sentence break
                if len(buffer) >= self.max_chunk_chars:
                    split = self._find_split_point(buffer, self.max_chunk_chars)
                    chunks.append(self._make_chunk(
                        buffer[:split].strip(), filename, page.page_num,
                        section, "text", buffer_start_bbox,
                        layout_tree_path=list(heading_stack),
                    ))
                    overlap_start = max(0, split - self.overlap)
                    buffer = buffer[overlap_start:]
                    buffer_start_bbox = None

        # Flush remaining
        if buffer.strip():
            chunks.append(self._make_chunk(
                buffer.strip(), filename, page.page_num,
                section, "text", buffer_start_bbox,
                layout_tree_path=list(heading_stack),
            ))

        return chunks

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_stub_headings(chunks: list[Chunk]) -> list[Chunk]:
        """Merge very short leading chunks into the next one.

        e.g. "填空题" → merge into "1．[图]，则[图] ."
        """
        if len(chunks) <= 1:
            return chunks
        merged: list[Chunk] = []
        pending: str | None = None
        for c in chunks:
            content = c.content.strip()
            # Very short chunk that's just a heading prefix
            if pending is None and len(content) <= 10:
                pending = content
                continue
            if pending:
                c = Chunk(
                    content=pending + "\n" + c.content,
                    metadata=c.metadata,
                )
                pending = None
            merged.append(c)
        # If the very last chunk is the stub, keep it as-is
        if pending:
            merged.append(chunks[-1])
        return merged

    @staticmethod
    def _merge_option_orphans(chunks: list[Chunk]) -> list[Chunk]:
        """Merge orphan option lines with preceding chunk.

        Handles:
        - Standard: ``"A．1； B．2； C．3； D．4"``
        - Image options: ``"[图]；B．[图]；C．[图]；D．[图]．"`` (A is an image)
        """
        if len(chunks) <= 1:
            return chunks

        merged: list[Chunk] = []
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            # Merge consecutive option-only chunks into the preceding chunk
            while i + 1 < len(chunks):
                nxt = chunks[i + 1]
                if nxt.metadata is None or nxt.metadata.chunk_type not in ("text",):
                    break
                first_line = nxt.content.split("\n", 1)[0]
                if not _OPTIONS_ONLY_RE.match(first_line):
                    break
                # Looks like orphan options — merge
                cur = Chunk(
                    content=cur.content + " " + nxt.content,
                    metadata=chunks[i].metadata,
                )
                i += 1  # skip merged
            merged.append(cur)
            i += 1
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_split_point(self, text: str, max_len: int) -> int:
        """在 max_len 附近找到合适的切分点（优先段末对齐）。"""
        if len(text) <= max_len:
            return len(text)
        candidate = max_len
        for boundary in ["\n\n", "\n", "。", "！", "？"]:
            pos = text.rfind(boundary, 0, max_len)
            if pos > max_len // 2:
                candidate = max(candidate, pos + len(boundary))
                break
        return min(candidate, max_len)

    def _make_chunk(
        self, content: str, filename: str, page_num: int,
        section: str, chunk_type: str, bbox=None,
        layout_tree_path: list[str] | None = None,
        chunk_level: str = "paragraph",
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
                chunk_level=chunk_level,
            ),
        )
