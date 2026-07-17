"""Word document loader using python-docx + lxml XML traversal.

Extracts:
  - Paragraphs with heading-level detection (style + font-size heuristics)
  - **Math formulas** (OMML ``m:oMath``) — missing from python-docx ``paragraph.text``
  - **Text boxes / floating shapes** (``w:txbxContent``) — invisible to the high-level API
  - Tables as structured ``Block`` objects

Since .docx has no physical page concept, the entire document is treated as a
single logical page for pipeline compatibility.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

from .base import DocumentLoader
from src.domain import Document, Page, Block
from src.domain.enums import ProcessingStatus

# ---------------------------------------------------------------------------
# OOXML namespace constants
# ---------------------------------------------------------------------------
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

_NSMAP = {
    "w": _W_NS,
    "m": _M_NS,
}

# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------
_HEADING_LEVEL_MAP = {
    "Heading 1": "title",
    "Heading 2": "section_heading",
    "Heading 3": "section_heading",
    "Heading 4": "section_heading",
    "Heading 5": "section_heading",
    "Heading 6": "section_heading",
    "Title": "title",
    "Subtitle": "section_heading",
    "1": "title",
    "2": "section_heading",
    "3": "section_heading",
}

_CHARS_PER_PAGE_ESTIMATE = 3000
_DEFAULT_PAGE_WIDTH = 595  # A4 pt
_DEFAULT_PAGE_HEIGHT = 842  # A4 pt


# ---------------------------------------------------------------------------
# Low-level XML helpers
# ---------------------------------------------------------------------------


def _collect_text_from_element(element) -> str:
    """Walk *element* in document order collecting text from ``w:t`` and ``m:t``.

    This captures both regular paragraph text and OMML math formula text,
    which python-docx's ``paragraph.text`` silently drops.
    """
    parts: list[str] = []
    for child in element.iter():
        tag = child.tag
        if "}" not in tag:
            continue
        ns = tag[tag.index("{") + 1 : tag.index("}")]
        local = tag[tag.rindex("}") + 1 :]
        if local == "t" and ns in (_W_NS, _M_NS) and child.text:
            parts.append(child.text)
    return "".join(parts)


def _has_math(element) -> bool:
    """Return ``True`` if *element* contains OMML math formula elements."""
    return len(element.findall(".//m:oMath", _NSMAP)) > 0


def _extract_math_text(element) -> str:
    """Extract the linear plain-text form of all math formulas in *element*."""
    parts: list[str] = []
    for math_elem in element.findall(".//m:oMath", _NSMAP):
        formula_text = _collect_text_from_element(math_elem)
        if formula_text.strip():
            parts.append(formula_text.strip())
    return " ".join(parts)


def _find_textbox_paragraphs(body) -> list[str]:
    """Extract text from every ``w:txbxContent`` in the document body.

    Text boxes / floating shapes are anchored to drawings but their paragraph
    content is not surfaced by ``doc.paragraphs``.
    """
    texts: list[str] = []
    for txbx in body.findall(".//w:txbxContent", _NSMAP):
        for p in txbx.findall(".//w:p", _NSMAP):
            text = _collect_text_from_element(p).strip()
            if text:
                texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Word Loader
# ---------------------------------------------------------------------------


class WordLoader(DocumentLoader):
    """Word document loader.

    Extracts paragraphs (with headings), math formulas, text-box content,
    and tables.  Uses ``lxml`` XML traversal on top of python-docx to
    capture elements the high-level API misses.
    """

    def load(self, path: str) -> Document:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if filepath.suffix.lower() not in (".docx", ".doc"):
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        docx_doc = DocxDocument(str(filepath))

        document = Document(
            filename=filepath.name,
            file_path=str(filepath.resolve()),
            file_type="docx",
            total_pages=1,
        )

        page = Page(
            page_num=1,
            width=_DEFAULT_PAGE_WIDTH,
            height=_DEFAULT_PAGE_HEIGHT,
        )

        # ------------------------------------------------------------------
        # 1. Paragraphs (including those with embedded math formulas)
        # ------------------------------------------------------------------
        for order_idx, paragraph in enumerate(docx_doc.paragraphs):
            p_elem = paragraph._element  # lxml element

            # Detect math — if present, extract the full text that includes
            # formula content instead of relying on paragraph.text.
            if _has_math(p_elem):
                text = _collect_text_from_element(p_elem).strip()
                math_text = _extract_math_text(p_elem)
                block_type = "formula"
                # Use math text as content; keep full text in metadata
                text = math_text or text
            else:
                text = paragraph.text.strip()
                if not text:
                    continue
                style_name = paragraph.style.name if paragraph.style else "Normal"
                block_type = _HEADING_LEVEL_MAP.get(style_name, "text")

                # Font-size heuristic for heading classification
                font_size = self._first_font_size(paragraph)
                if block_type == "text" and font_size:
                    if font_size >= 18:
                        block_type = "title"
                    elif font_size >= 14:
                        block_type = "section_heading"

            if not text:
                continue

            metadata: dict = {}
            if paragraph.style:
                metadata["style"] = paragraph.style.name
            if _has_math(p_elem):
                metadata["has_formula"] = True

            block = Block(
                content=text,
                block_type=block_type,
                page_num=1,
                bbox=(0, 0, _DEFAULT_PAGE_WIDTH, 20),
                reading_order=order_idx,
                metadata=metadata,
            )
            page.blocks.append(block)
            page.text += text + "\n"

        # ------------------------------------------------------------------
        # 2. Text-box / floating-shape content
        # ------------------------------------------------------------------
        tb_texts = _find_textbox_paragraphs(docx_doc.element.body)
        tb_offset = len(docx_doc.paragraphs)
        for i, tb_text in enumerate(tb_texts):
            block = Block(
                content=tb_text,
                block_type="text",
                page_num=1,
                bbox=(0, 0, _DEFAULT_PAGE_WIDTH, 20),
                reading_order=tb_offset + i,
                metadata={"source": "textbox"},
            )
            page.blocks.append(block)
            page.text += tb_text + "\n"

        # ------------------------------------------------------------------
        # 3. Tables
        # ------------------------------------------------------------------
        table_offset = tb_offset + len(tb_texts)
        for table_idx, table in enumerate(docx_doc.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))

            table_text = "\n".join(rows)
            if not table_text.strip():
                continue

            block = Block(
                content=table_text,
                block_type="table",
                page_num=1,
                bbox=(0, 0, _DEFAULT_PAGE_WIDTH, 20 * len(rows)),
                reading_order=table_offset + table_idx,
                metadata={
                    "style": "Table",
                    "num_rows": len(table.rows),
                    "num_cols": len(table.columns) if table.columns else 0,
                },
            )
            page.blocks.append(block)
            page.text += table_text + "\n"

        document.pages.append(page)
        document.status = ProcessingStatus.UPLOADED

        estimated_pages = max(1, len(page.text) // _CHARS_PER_PAGE_ESTIMATE)
        document.total_pages = estimated_pages

        return document

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_font_size(paragraph) -> float | None:
        """Return the first non-``None`` font size (``pt``) in the paragraph."""
        for run in paragraph.runs:
            if run.font.size:
                return run.font.size.pt
        return None
