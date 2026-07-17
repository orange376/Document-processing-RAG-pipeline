from pathlib import Path
import fitz  # PyMuPDF

from .base import DocumentLoader
from src.domain import Document, Page, Block
from src.domain.enums import ProcessingStatus


class PDFLoader(DocumentLoader):
    """PDF 文档加载器 — 使用 PyMuPDF 提取文本块 + 版面元数据

    提取内容：
      - blocks: 用于切片的基础文本块（含字号信息）
      - raw_dict: 用于启发式版面分析的 PyMuPDF dict 输出
    """

    def load(self, path: str) -> Document:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if filepath.suffix.lower() != ".pdf":
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        doc = fitz.open(path)
        document = Document(
            filename=filepath.name,
            file_path=str(filepath.resolve()),
            file_type="pdf",
            total_pages=len(doc),
        )

        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect

            pdf_page = Page(
                page_num=page_num + 1,
                width=rect.width,
                height=rect.height,
            )

            # ── 提取详细 dict（含字号、字体、位置） ──
            raw = page.get_text("dict")
            pdf_page.raw_dict = raw

            # ── 提取文本块 ──
            for order_idx, block in enumerate(raw.get("blocks", [])):
                block_type = block.get("type", 0)

                if block_type == 0:  # 文本块
                    bbox = block.get("bbox", (0, 0, 0, 0))
                    text_parts = []
                    max_font_size = 0.0
                    font_name = ""

                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_parts.append(span.get("text", ""))
                            size = span.get("size", 11)
                            if size > max_font_size:
                                max_font_size = size
                            if not font_name:
                                font_name = span.get("font", "")

                    full_text = "".join(text_parts).strip()
                    if not full_text:
                        continue

                    block_obj = Block(
                        content=full_text,
                        block_type="text",
                        page_num=page_num + 1,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        reading_order=order_idx,
                        metadata={
                            "max_font_size": max_font_size,
                            "font": font_name,
                            "has_formula": self._is_formula(full_text),
                        },
                    )
                    pdf_page.blocks.append(block_obj)
                    pdf_page.text += full_text + "\n"

                elif block_type == 1:  # 图片块
                    bbox = block.get("bbox", (0, 0, 0, 0))
                    # 提取图像数据
                    img_bytes = block.get("image", None)
                    block_obj = Block(
                        content="",
                        block_type="image",
                        page_num=page_num + 1,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        reading_order=order_idx,
                    )
                    pdf_page.blocks.append(block_obj)
                    if img_bytes:
                        pdf_page.images.append(img_bytes)

            document.pages.append(pdf_page)

        doc.close()
        document.status = ProcessingStatus.UPLOADED
        return document

    def _is_formula(self, text: str) -> bool:
        """启发式检测公式（含常见数学符号）"""
        math_symbols = {"∑", "∫", "∂", "√", "π", "∞", "Δ", "λ", "→", "≈", "∈", "⊂"}
        return any(s in text for s in math_symbols)
