from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import get_settings
from src.domain import Document, Chunk, ProcessingStatus
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
        self._layout_detector = LayoutDetector()
        for page in document.pages:
            try:
                # Try PP-DocLayoutV3 on rendered page image
                img = self._page_to_array(page, str(path), scale=self.RENDER_SCALE)
                elements = self._layout_detector.analyze(
                    img, scale=self.RENDER_SCALE
                )
                if elements:
                    page.layout_elements = elements
                    continue
            except Exception:
                pass

            # Fallback: heuristic from PyMuPDF dict blocks
            if page.raw_dict:
                page.layout_elements = self._layout_detector.analyze_from_blocks(
                    page.raw_dict.get("blocks", []),
                    page.page_num,
                    page.width,
                    page.height,
                )
            else:
                page.layout_elements = []
        self._layout_detector.unload()
        self._layout_detector = None

        # === 阶段 3: OCR（仅对缺文本页面执行） ===
        pages_needing_ocr = [p for p in document.pages if not p.text.strip()]
        if pages_needing_ocr:
            try:
                self._ocr_engine = OCREngine()
                for page in pages_needing_ocr:
                    image = self._page_to_array(page, str(path), scale=2.0)
                    self._ocr_engine.recognize(image, page)
                self._ocr_engine.unload()
                self._ocr_engine = None
            except Exception:
                self._ocr_engine = None

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
            pass

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
        self._layout_detector = LayoutDetector()
        for page in document.pages:
            try:
                img = self._page_to_array(page, str(path), scale=self.RENDER_SCALE)
                elements = self._layout_detector.analyze(img, scale=self.RENDER_SCALE)
                if elements:
                    page.layout_elements = elements
                    continue
            except Exception:
                pass
            if page.raw_dict:
                page.layout_elements = self._layout_detector.analyze_from_blocks(
                    page.raw_dict.get("blocks", []),
                    page.page_num, page.width, page.height,
                )
            else:
                page.layout_elements = []
        self._layout_detector.unload()
        self._layout_detector = None

        # === 阶段 3: OCR ===
        pages_needing_ocr = [p for p in document.pages if not p.text.strip()]
        if pages_needing_ocr:
            try:
                self._ocr_engine = OCREngine()
                for page in pages_needing_ocr:
                    image = self._page_to_array(page, str(path), scale=2.0)
                    self._ocr_engine.recognize(image, page)
                self._ocr_engine.unload()
                self._ocr_engine = None
            except Exception:
                self._ocr_engine = None

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
    ) -> np.ndarray:
        """将 Page 渲染为 numpy 数组

        Args:
            page: 领域层 Page 对象
            file_path: PDF 文件路径（用于实际渲染），为 None 时返回空白图
            scale: 渲染倍率（2x = 144 DPI）

        Returns:
            (H, W, 3) uint8 numpy 数组
        """
        import fitz

        if file_path:
            try:
                doc = fitz.open(file_path)
                pdf_page = doc[page.page_num - 1]
                mat = fitz.Matrix(scale, scale)
                pix = pdf_page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                doc.close()
                return img[:, :, :3]  # ensure RGB
            except Exception:
                pass
        return np.zeros((800, 800, 3), dtype=np.uint8)
