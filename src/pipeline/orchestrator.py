from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)
from src.domain import Document, Page, Block, Chunk, ProcessingStatus
from src.domain import BBox, LayoutElement
from src.parser.loader.pdf_loader import PDFLoader
from src.parser.loader.word_loader import WordLoader
from src.parser.layout.detector import LayoutDetector
from src.parser.ocr.engine import OCREngine
from src.parser.layout_tree import LayoutTreeBuilder, LayoutTreeNode
from src.parser.chunker import StructureAwareChunker
from src.index.embedding import EmbeddingEngine
from src.index.vector_store import VectorStore


@dataclass
class ProcessingResult:
    """文档处理结果"""
    document: Document
    chunks: list[Chunk]
    layout_tree: LayoutTreeNode | None = None
    confidence: float = 0.0
    status: ProcessingStatus = ProcessingStatus.PROCESSING
    indexed_count: int = 0


class PipelineOrchestrator:
    """文档处理流水线主编排器

    流程:
      1. PDF 加载（PyMuPDF）
      2. 版面分析 — PP-DocLayoutV3（25 类深度学习模型）
      3. OCR — 仅对缺少文本的页面执行（easyocr）
      4. 版面树构建
      5. 结构感知切片
      6. Embedding + Qdrant 索引
    """

    RENDER_SCALE = 2.0  # PyMuPDF render scale for layout analysis

    def __init__(self):
        self._settings = get_settings()
        self._vector_store = VectorStore()
        self._layout_detector: LayoutDetector | None = None
        self._ocr_engine: OCREngine | None = None
        self._embedding_engine: EmbeddingEngine | None = None

    async def process_document(self, file_path: str) -> ProcessingResult:
        """全流程处理一个文档

        Args:
            file_path: 文档的本地路径

        Returns:
            ProcessingResult 包含处理结果
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # === 阶段 1: 加载文档 ===
        loader = self._get_loader(path)
        document = loader.load(str(path))
        document.status = ProcessingStatus.PROCESSING

        # === 阶段 2: 版面分析（PP-DocLayoutV3 + 启发式降级） ===
        is_pdf = path.suffix.lower() == ".pdf"
        self._layout_detector = LayoutDetector() if is_pdf else None
        for page in document.pages:
            # --- PDF: try PP-DocLayoutV3 on rendered page image ---
            img = self._page_to_array(page, str(path), scale=self.RENDER_SCALE)
            if img is not None and self._layout_detector:
                try:
                    elements = self._layout_detector.analyze(img, scale=self.RENDER_SCALE)
                    if elements:
                        page.layout_elements = elements
                        continue
                except Exception:
                    pass

            # --- Fallback ---
            if page.raw_dict:
                # PDF heuristic from PyMuPDF dict (two-column, font-size)
                if self._layout_detector is None:
                    self._layout_detector = LayoutDetector()
                page.layout_elements = self._layout_detector.analyze_from_blocks(
                    page.raw_dict.get("blocks", []),
                    page.page_num,
                    page.width,
                    page.height,
                )
            elif path.suffix.lower() in (".docx", ".doc"):
                # Word: use pre-classified block metadata
                page.layout_elements = self._build_layout_from_word_blocks(page)
            else:
                page.layout_elements = []

        if self._layout_detector:
            self._layout_detector.unload()
        self._layout_detector = None

        # === 阶段 3: OCR（仅对缺文本页面执行） ===
        pages_needing_ocr = [p for p in document.pages if not p.text.strip()]
        has_embedded_images = any(p.images for p in document.pages)

        if pages_needing_ocr or has_embedded_images:
            try:
                self._ocr_engine = OCREngine()
                for page in pages_needing_ocr:
                    image = self._page_to_array(page, str(path), scale=2.0)
                    if image is not None:
                        self._ocr_engine.recognize(image, page)

                # OCR for embedded images (Word formula screenshots, etc.)
                if has_embedded_images:
                    await self._ocr_embedded_images(document)

                self._ocr_engine.unload()
                self._ocr_engine = None
            except Exception:
                self._ocr_engine = None

        # --- Scanned PDF page recognition via Qwen-VL ---
        # Pages that have no extractable text AND contain embedded images are
        # likely scanned document pages — use multimodal LLM for full-page OCR.
        if path.suffix.lower() == ".pdf":
            scanned_pages = [
                p for p in document.pages
                if not p.text.strip() and len(p.images) > 0
            ]
            if scanned_pages:
                try:
                    from src.ocr import PageRecognizer
                    page_recognizer = PageRecognizer()
                    for sp in scanned_pages:
                        img = self._page_to_array(sp, str(path), scale=1.5)
                        if img is None:
                            continue
                        markdown, conf = await page_recognizer.recognize_page(img)
                        if markdown:
                            sp.text = markdown
                            sp.blocks = [Block(
                                content=markdown,
                                block_type="text",
                                page_num=sp.page_num,
                                bbox=(0, 0, sp.width, sp.height),
                                reading_order=0,
                                confidence=conf,
                                metadata={"source": "qwen_vl_page_recognizer"},
                            )]
                            logger.info(
                                "Qwen-VL page %d → %d chars (conf=%.2f)",
                                sp.page_num, len(markdown), conf,
                            )
                except Exception:
                    logger.exception("Scanned page recognition (Qwen-VL) failed")

        # === 阶段 4: 构建版面树 ===
        tree_builder = LayoutTreeBuilder()
        all_elements = []
        for page in document.pages:
            all_elements.extend(page.layout_elements)
        layout_tree = tree_builder.build(all_elements)

        # === 阶段 5: 结构感知切片 ===
        chunker = StructureAwareChunker()
        chunks = chunker.chunk(document)

        # === 阶段 6: Embedding + 索引 ===
        indexed = 0
        try:
            self._embedding_engine = EmbeddingEngine()
            chunks = self._embedding_engine.embed_chunks(chunks)
            self._embedding_engine.unload()
            self._embedding_engine = None
            indexed = self._vector_store.index_chunks(chunks)
        except Exception:
            logger.exception("Embedding / indexing failed for %s", path)
            indexed = 0

        status = ProcessingStatus.INDEXED if indexed > 0 else ProcessingStatus.PROCESSING
        document.status = status
        return ProcessingResult(
            document=document,
            chunks=chunks,
            layout_tree=layout_tree,
            status=status,
            indexed_count=indexed,
        )

    async def parse_document(self, file_path: str) -> ProcessingResult:
        """纯解析文档（load → layout → OCR → tree → chunk），不做索引。

        Args:
            file_path: 文档的本地路径

        Returns:
            ProcessingResult（indexed_count 始终为 0）
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # === 阶段 1: 加载文档 ===
        loader = self._get_loader(path)
        document = loader.load(str(path))
        document.status = ProcessingStatus.PROCESSING

        # === 阶段 2: 版面分析 ===
        is_pdf = path.suffix.lower() == ".pdf"
        self._layout_detector = LayoutDetector() if is_pdf else None
        for page in document.pages:
            img = self._page_to_array(page, str(path), scale=self.RENDER_SCALE)
            if img is not None and self._layout_detector:
                try:
                    elements = self._layout_detector.analyze(img, scale=self.RENDER_SCALE)
                    if elements:
                        page.layout_elements = elements
                        continue
                except Exception:
                    pass
            if page.raw_dict:
                if self._layout_detector is None:
                    self._layout_detector = LayoutDetector()
                page.layout_elements = self._layout_detector.analyze_from_blocks(
                    page.raw_dict.get("blocks", []),
                    page.page_num, page.width, page.height,
                )
            elif path.suffix.lower() in (".docx", ".doc"):
                page.layout_elements = self._build_layout_from_word_blocks(page)
            else:
                page.layout_elements = []
        if self._layout_detector:
            self._layout_detector.unload()
        self._layout_detector = None

        # === 阶段 3: OCR ===
        pages_needing_ocr = [p for p in document.pages if not p.text.strip()]
        has_embedded_images = any(p.images for p in document.pages)

        if pages_needing_ocr or has_embedded_images:
            try:
                self._ocr_engine = OCREngine()
                for page in pages_needing_ocr:
                    image = self._page_to_array(page, str(path), scale=2.0)
                    if image is not None:
                        self._ocr_engine.recognize(image, page)

                # OCR for embedded images (Word formula screenshots, etc.)
                if has_embedded_images:
                    await self._ocr_embedded_images(document)

                self._ocr_engine.unload()
                self._ocr_engine = None
            except Exception:
                self._ocr_engine = None

        # --- Scanned PDF page recognition via Qwen-VL ---
        if path.suffix.lower() == ".pdf":
            scanned_pages = [
                p for p in document.pages
                if not p.text.strip() and len(p.images) > 0
            ]
            if scanned_pages:
                try:
                    from src.ocr import PageRecognizer
                    page_recognizer = PageRecognizer()
                    for sp in scanned_pages:
                        img = self._page_to_array(sp, str(path), scale=1.5)
                        if img is None:
                            continue
                        markdown, conf = await page_recognizer.recognize_page(img)
                        if markdown:
                            sp.text = markdown
                            sp.blocks = [Block(
                                content=markdown,
                                block_type="text",
                                page_num=sp.page_num,
                                bbox=(0, 0, sp.width, sp.height),
                                reading_order=0,
                                confidence=conf,
                                metadata={"source": "qwen_vl_page_recognizer"},
                            )]
                            logger.info(
                                "Qwen-VL page %d → %d chars (conf=%.2f)",
                                sp.page_num, len(markdown), conf,
                            )
                except Exception:
                    logger.exception("Scanned page recognition (Qwen-VL) failed")

        # === 阶段 4: 构建版面树 ===
        tree_builder = LayoutTreeBuilder()
        all_elements = []
        for page in document.pages:
            all_elements.extend(page.layout_elements)
        layout_tree = tree_builder.build(all_elements)

        # === 阶段 5: 结构感知切片 ===
        chunker = StructureAwareChunker()
        chunks = chunker.chunk(document)

        return ProcessingResult(
            document=document,
            chunks=chunks,
            layout_tree=layout_tree,
            status=ProcessingStatus.PROCESSING,
        )

    async def _ocr_embedded_images(self, document: Document) -> None:
        """Run OCR / formula recognition on embedded images.

        Images are now tracked via ``[IMG_N]`` placeholders placed at the
        correct text position by :class:`WordLoader`.

        For each image:
          1. Try :class:`FormulaRecognizer` (Qwen-VL) — if it returns LaTeX,
             replace the placeholder with ``$$...$$``.
          2. Otherwise fall back to easyocr and replace with plain text.
        """
        import io

        import numpy as np
        from PIL import Image
        from src.ocr import FormulaRecognizer

        formula_recognizer = FormulaRecognizer()

        for page in document.pages:
            if not page.images:
                continue
            for block in page.blocks:
                img_indices: list[int] = block.metadata.get(
                    "embedded_image_indices", []
                )
                if not img_indices:
                    continue

                for idx in img_indices:
                    if idx >= len(page.images):
                        continue

                    placeholder = f"[IMG_{idx}]"
                    if placeholder not in block.content:
                        continue  # sanity check — should always match

                    try:
                        img_bytes = page.images[idx]
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        img_array = np.array(pil_img.convert("RGB"))

                        # --- Step 1: Try formula recognition (Qwen-VL) ---
                        latex, formula_conf = await formula_recognizer.recognize(
                            img_bytes
                        )

                        if latex:
                            block.content = block.content.replace(
                                placeholder, latex, 1
                            )
                            continue  # done for this image

                        # --- Step 2: Fall back to easyocr ---
                        ocr_text = self._ocr_engine.recognize(img_array)
                        ocr_clean = ocr_text.strip() if ocr_text else "[无法识别]"

                        block.content = block.content.replace(
                            placeholder, ocr_clean, 1
                        )

                    except Exception:
                        # Replace with a visible marker on failure
                        block.content = block.content.replace(
                            placeholder, "[图片识别失败]", 1
                        )

    @staticmethod
    def _build_layout_from_word_blocks(page: Page) -> list[LayoutElement]:
        """从 WordLoader 已分类的 Block 构建 LayoutElement 列表。

        WordLoader 通过样式名、字号、内容启发式机制将 block 分类为
        title/heading/formula/table/text 等类型。这里直接将这些
        已分类的信息转换为 LayoutElement，使 Word 文档的版面树和
        置信度评估也能正常工作（Word 无法渲染为页面图片走 PP-DocLayoutV3）。
        """
        BLOCK_TO_CATEGORY = {
            "title": "title",
            "heading": "section_heading",
            "section_heading": "section_heading",
            "text": "text",
            "formula": "formula",
            "table": "table",
            "figure": "figure",
            "image": "figure",
        }

        elements: list[LayoutElement] = []
        for block in page.blocks:
            category = BLOCK_TO_CATEGORY.get(block.block_type, "text")
            x0, y0, x1, y1 = block.bbox if block.bbox else (0, 0, page.width, 20)
            elements.append(LayoutElement(
                bbox=BBox(
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    page_num=block.page_num,
                ),
                category=category,
                confidence=block.confidence or 0.85,
                reading_order=block.reading_order,
                text=block.content[:200],
            ))

        return elements

    def _get_loader(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PDFLoader()
        elif suffix in (".docx", ".doc"):
            return WordLoader()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def _page_to_array(
        self, page, file_path: str | None = None, scale: float = 2.0
    ) -> np.ndarray | None:
        """将 Page 渲染为 numpy 数组

        仅对 PDF 文件做实际渲染（PyMuPDF）。
        Word 等无原生页面渲染能力的格式返回 ``None``，
        由调用方切换到对应的降级路径。

        Args:
            page: 领域层 Page 对象
            file_path: PDF 文件路径（用于实际渲染）
            scale: 渲染倍率（2x = 144 DPI）

        Returns:
            (H, W, 3) uint8 numpy 数组，或 None（无法渲染时）
        """
        path = Path(file_path) if file_path else None
        if path and path.suffix.lower() == ".pdf":
            return self._render_pdf_page(page, file_path, scale)
        return None

    def _render_pdf_page(
        self, page, file_path: str, scale: float = 2.0
    ) -> np.ndarray | None:
        """用 PyMuPDF 渲染 PDF 页面到 numpy 数组"""
        import fitz

        try:
            doc = fitz.open(file_path)
            pdf_page = doc[page.page_num - 1]
            mat = fitz.Matrix(scale, scale)
            pix = pdf_page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            doc.close()
            return img[:, :, :3]
        except Exception:
            return None
