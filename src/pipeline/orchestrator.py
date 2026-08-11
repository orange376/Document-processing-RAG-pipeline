from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import get_settings

logger = logging.getLogger(__name__)

from src.domain import BBox, Block, Chunk, Document, LayoutElement, Page, ProcessingStatus
from src.index.embedding import EmbeddingEngine
from src.index.vector_store import VectorStore
from src.parser.chunker import StructureAwareChunker
from src.parser.layout.detector import LayoutDetector
from src.parser.layout_tree import LayoutTreeBuilder, LayoutTreeNode
from src.parser.loader.pdf_loader import PDFLoader
from src.parser.loader.word_loader import WordLoader
from src.parser.ocr.engine import OCREngine

# Qwen-VL bills image input by resolution and latency grows with upload size.
_MAX_VL_IMAGE_EDGE = 1024  # cap the longest edge before sending to Qwen-VL


def _encode_vl_image(pil_img: Image.Image) -> bytes:
    """Downsample + JPEG-encode a crop for Qwen-VL.

    Sending full-res PNG crops is both slower (bigger upload) and more
    expensive (image tokens scale with resolution). Capping the longest edge
    at 1024px and using JPEG keeps diagram legibility while cutting both.
    """
    import io

    w, h = pil_img.size
    if max(w, h) > _MAX_VL_IMAGE_EDGE:
        scale = _MAX_VL_IMAGE_EDGE / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@dataclass
class ProcessingResult:
    """文档处理结果"""
    document: Document
    chunks: list[Chunk]
    layout_tree: LayoutTreeNode | None = None
    confidence: float = 0.0
    status: ProcessingStatus = ProcessingStatus.PROCESSING
    indexed_count: int = 0


def _bbox_overlap(a, b) -> bool:
    """Return True if two bboxes have any overlap (IoU or containment).

    Handles :class:`BBox` objects (with .x0/.y0/.x1/.y1) and plain tuples.
    Returns False when either bbox is None.
    """
    if a is None or b is None:
        return False
    try:
        ax0, ay0, ax1, ay1 = _unpack_bbox(a)
        bx0, by0, bx1, by1 = _unpack_bbox(b)
        # Check if they don't overlap
        if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
            return False
        return True
    except Exception:
        return False


def _unpack_bbox(b):
    """Unpack a BBox (dataclass) or tuple into (x0, y0, x1, y1)."""
    if hasattr(b, "x0"):
        return (b.x0, b.y0, b.x1, b.y1)
    return tuple(b)


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
        import time

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        t_total = time.perf_counter()

        # === 阶段 1: 加载文档 ===
        t0 = time.perf_counter()
        loader = self._get_loader(path)
        document = loader.load(str(path))
        document.status = ProcessingStatus.PROCESSING
        logger.info("[stage] load: %.2fs (pages=%d)", time.perf_counter() - t0, document.total_pages)

        # === 阶段 2: 版面分析（PP-DocLayoutV3 + 启发式降级） ===
        t0 = time.perf_counter()
        is_pdf = path.suffix.lower() == ".pdf"
        self._layout_detector = LayoutDetector() if is_pdf else None
        total_pages = len(document.pages)
        logger.info("[stage] layout: starting (pages=%d)", total_pages)
        for pi, page in enumerate(document.pages):
            # --- PDF: try PP-DocLayoutV3 on rendered page image ---
            render_t0 = time.perf_counter()
            img = self._page_to_array(page, str(path), scale=self.RENDER_SCALE)
            render_elapsed = time.perf_counter() - render_t0
            if img is not None and self._layout_detector:
                try:
                    elements = self._layout_detector.analyze(img, scale=self.RENDER_SCALE)
                    if elements:
                        page.layout_elements = elements
                        logger.info(
                            "[stage] layout: page %d/%d (render=%.2fs, elements=%d)",
                            pi + 1, total_pages, render_elapsed, len(elements),
                        )
                        continue
                except Exception:
                    logger.warning("[stage] layout: page %d/%d FAILED, falling back", pi + 1, total_pages)

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
        logger.info("[stage] layout_analysis: %.2fs", time.perf_counter() - t0)

        # === 阶段 2.5: 图表描述 + 表格结构恢复（Qwen-VL） ===
        logger.info("[stage] describe: starting (figures + tables)")
        # Figure description + table recovery are independent Qwen-VL workloads
        # over disjoint regions — run them concurrently (both are API-latency bound).
        await asyncio.gather(
            self._describe_figures(document, str(path)),
            self._recover_tables(document, str(path)),
        )

        # === 阶段 3: OCR（仅对缺文本页面执行） ===
        t0 = time.perf_counter()
        pages_needing_ocr = [p for p in document.pages if not p.text.strip()]
        has_embedded_images = any(p.images for p in document.pages)
        logger.info(
            "[stage] ocr: starting (need_ocr=%d, embedded_images=%s)",
            len(pages_needing_ocr), has_embedded_images,
        )

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
        logger.info("[stage] ocr: %.2fs", time.perf_counter() - t0)

        # === 阶段 4: 构建版面树 ===
        logger.info("[stage] tree_build: starting")
        t0 = time.perf_counter()
        tree_builder = LayoutTreeBuilder()
        all_elements = []
        for page in document.pages:
            all_elements.extend(page.layout_elements)
        layout_tree = tree_builder.build(all_elements)
        logger.info("[stage] tree_build: %.2fs (elements=%d)", time.perf_counter() - t0, len(all_elements))

        # === 阶段 5: 结构感知切片（多粒度：段落 + 句子） ===
        logger.info("[stage] chunk: starting")
        t0 = time.perf_counter()
        chunker = StructureAwareChunker()
        chunks = chunker.fine_chunk(document)
        para_count = sum(1 for c in chunks if c.metadata and c.metadata.chunk_level in ("paragraph", ""))
        sent_count = sum(1 for c in chunks if c.metadata and c.metadata.chunk_level == "sentence")
        logger.info(
            "[stage] chunk: %.2fs (total=%d, paragraph=%d, sentence=%d)",
            time.perf_counter() - t0, len(chunks), para_count, sent_count,
        )

        # === 阶段 6: Embedding + 索引（CPU 密集型，放入线程池） ===

        indexed = 0
        try:
            logger.info("[stage] embed+index: starting (chunks=%d)", len(chunks))
            t0 = time.perf_counter()
            self._embedding_engine = EmbeddingEngine()
            chunks = await asyncio.to_thread(self._embedding_engine.embed_chunks, chunks)
            self._embedding_engine.unload()
            self._embedding_engine = None
            indexed = await asyncio.to_thread(self._vector_store.index_chunks, chunks)
            logger.info("[stage] embed+index: %.2fs (indexed=%d)", time.perf_counter() - t0, indexed)
        except Exception:
            logger.exception("Embedding / indexing failed for %s", path)
            indexed = 0

        status = ProcessingStatus.INDEXED if indexed > 0 else ProcessingStatus.PROCESSING
        document.status = status
        logger.info("[stage] TOTAL: %.2fs (file=%s)", time.perf_counter() - t_total, path.name)
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

        # === 阶段 2.5: 表格结构恢复（Qwen-VL） ===
        await self._recover_tables(document, str(path))

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

        # === 阶段 5: 结构感知切片（多粒度） ===
        chunker = StructureAwareChunker()
        chunks = chunker.fine_chunk(document)

        return ProcessingResult(
            document=document,
            chunks=chunks,
            layout_tree=layout_tree,
            status=ProcessingStatus.PROCESSING,
        )

    async def _ocr_embedded_images(self, document: Document) -> None:
        """Run OCR / formula / table recognition on embedded images, routed by type.

        Images are tracked via ``[IMG_N]`` placeholders placed at the
        correct text position by :class:`WordLoader`.

        Uses asyncio.gather() for concurrent batch processing —
        116 images complete in seconds instead of minutes.

        Routing strategy:
          - ``block_type == "formula"``  → :class:`LatexOCREngine` (pix2tex, primary) →
            :class:`FormulaRecognizer` (Qwen-VL → LaTeX, fallback)
          - ``block_type == "table"``    → Qwen-VL table structure → markdown table
          - ``block_type == "figure"`` or other → Qwen-VL general → easyocr fallback
        """

        import numpy as np

        from src.ocr import FormulaRecognizer

        formula_recognizer = FormulaRecognizer()

        # 本地公式识别（pix2tex）—— 优先使用，失败回退 Qwen-VL
        from src.ocr import LatexOCREngine
        latex_ocr: LatexOCREngine | None = None
        try:
            latex_ocr = LatexOCREngine()
        except Exception:
            latex_ocr = None

        # Collect all (block, img_idx, img_bytes, block_category) tasks
        tasks: list[tuple] = []
        for page in document.pages:
            if not page.images:
                continue
            for block in page.blocks:
                img_indices: list[int] = block.metadata.get(
                    "embedded_image_indices", []
                )
                if not img_indices:
                    continue

                # Determine routing category
                block_category = block.block_type
                for le in page.layout_elements:
                    if le.category in ("table", "formula", "figure") and _bbox_overlap(
                        le.bbox, block.bbox
                    ):
                        block_category = le.category
                        break

                for idx in img_indices:
                    if idx >= len(page.images):
                        continue
                    placeholder = f"[IMG_{idx}]"
                    if placeholder not in block.content:
                        continue
                    tasks.append((block, idx, page.images[idx], placeholder, block_category))

        if not tasks:
            return

        total = len(tasks)
        logger.info("Embedded image recognition: %d images (concurrent)", total)

        completed = 0

        async def _recognize_one(
            block, idx: int, img_bytes: bytes, placeholder: str, category: str
        ) -> None:
            nonlocal completed
            try:
                # --- Route by image/block type ---
                if category == "formula":
                    # 主路径：本地 pix2tex
                    if latex_ocr is not None:
                        latex, _ = await asyncio.to_thread(
                            latex_ocr.recognize, img_bytes
                        )
                        if latex:
                            block.content = block.content.replace(
                                placeholder, latex, 1
                            )
                            completed += 1
                            return
                    # 回退：Qwen-VL
                    latex, _ = await formula_recognizer.recognize(img_bytes)
                    if latex:
                        block.content = block.content.replace(
                            placeholder, latex, 1
                        )
                        completed += 1
                        return
                    # 两个公式引擎（pix2tex + Qwen-VL）均未识别成功。
                    # 在此返回，避免落入下面的通用分支导致重复调用
                    # formula_recognizer（重复的 Qwen-VL API 调用）。
                    return

                elif category == "table":
                    table_md = await self._recognize_table(img_bytes)
                    if table_md:
                        block.content = block.content.replace(
                            placeholder, "\n" + table_md + "\n", 1
                        )
                        completed += 1
                        return

                # --- General: try formula first, then easyocr ---
                latex, _ = await formula_recognizer.recognize(img_bytes)
                if latex:
                    block.content = block.content.replace(placeholder, latex, 1)
                    completed += 1
                    return

                import io as _io

                from PIL import Image as _Image

                pil_img = _Image.open(_io.BytesIO(img_bytes))
                img_array = np.array(pil_img.convert("RGB"))
                if self._ocr_engine:
                    ocr_text = self._ocr_engine.recognize(img_array)
                    block.content = block.content.replace(
                        placeholder, ocr_text.strip() or "[无法识别]", 1
                    )
                else:
                    block.content = block.content.replace(
                        placeholder, "[无法识别]", 1
                    )
                completed += 1
            except Exception:
                block.content = block.content.replace(placeholder, "[图片识别失败]", 1)
                completed += 1

        # Run in batches to avoid overwhelming the Qwen-VL API
        BATCH = 5  # concurrent API calls
        for batch_start in range(0, len(tasks), BATCH):
            batch = tasks[batch_start:batch_start + BATCH]
            await asyncio.gather(*[
                _recognize_one(block, idx, img_bytes, placeholder, cat)
                for block, idx, img_bytes, placeholder, cat in batch
            ])
            logger.info(
                "Embedded images: %d/%d recognized (batch %d-%d)",
                completed, total, batch_start + 1, min(batch_start + BATCH, total),
            )

    async def _describe_figures(self, document: Document, file_path: str) -> None:
        """Describe figure/diagram regions via Qwen-VL and insert as text blocks.

        Covers: architecture diagrams, flowcharts, use-case diagrams, charts,
        graphs, and any other non-text visual content detected by layout analysis.

        Only runs for PDFs (Word embedded images are handled by
        :meth:`_ocr_embedded_images`).
        """

        import numpy as np

        from src.domain import Block

        if not file_path.lower().endswith(".pdf"):
            return

        # Collect figure regions
        figure_tasks: list[tuple] = []
        for page in document.pages:
            figure_elements = [
                le for le in page.layout_elements
                if le.category.lower() in ("figure", "image", "chart", "picture")
            ]
            if not figure_elements:
                continue

            page_img = self._page_to_array(page, file_path, scale=1.5)
            if page_img is None:
                continue
            page_h, page_w = page_img.shape[:2]
            scale_x = page_w / page.width
            scale_y = page_h / page.height

            for fe in figure_elements:
                try:
                    x0, y0, x1, y1 = _unpack_bbox(fe.bbox)
                    px0 = max(0, int(x0 * scale_x))
                    py0 = max(0, int(y0 * scale_y))
                    px1 = min(page_w, int(x1 * scale_x))
                    py1 = min(page_h, int(y1 * scale_y))
                    if px1 <= px0 or py1 <= py0:
                        continue
                    crop = page_img[py0:py1, px0:px1]
                    pil_img = Image.fromarray(crop.astype(np.uint8))
                    figure_tasks.append((page, fe, _encode_vl_image(pil_img), (x0, y0, x1, y1)))
                except Exception:
                    continue

        if not figure_tasks:
            return

        total = len(figure_tasks)
        logger.info("Figure description: %d regions (concurrent)", total)
        completed = 0

        async def _describe_one(page, fe, img_bytes, bbox_coords) -> None:
            nonlocal completed
            try:
                from src.generation.llm_client import LLMClient

                if not get_settings().qwen_api_key:
                    completed += 1
                    return

                prompt = (
                    "请详细描述这张图片的内容。"
                    "如果是架构图、流程图、用例图、ER图或图表，"
                    "请用中文仔细描述图中的节点、箭头、关系、层次结构、"
                    "数据流向等，使读者能够仅凭文字完全理解图表含义。"
                    "如果是普通图片或照片，请简要描述内容。"
                    "请直接用中文回复，不要加前缀。"
                )
                description = await LLMClient(provider="qwen").chat_with_image(
                    prompt,
                    img_bytes,
                    system="你是一个图表理解专家。将图片内容用中文详细描述。",
                )

                if description and len(description.strip()) > 10:
                    x0, y0, x1, y1 = bbox_coords
                    block = Block(
                        content=f"[图表描述] {description.strip()}",
                        block_type="figure",
                        page_num=page.page_num,
                        bbox=(x0, y0, x1, y1),
                        reading_order=fe.reading_order or 0,
                        confidence=fe.confidence,
                        metadata={"source": "qwen_vl_figure_describe"},
                    )
                    page.blocks.append(block)
                    page.text += description + "\n"
                completed += 1
            except Exception:
                completed += 1

        BATCH = 5  # concurrent Qwen-VL calls (429 handled by retry)
        for batch_start in range(0, len(figure_tasks), BATCH):
            batch = figure_tasks[batch_start:batch_start + BATCH]
            await asyncio.gather(*[
                _describe_one(page, fe, img_bytes, bbox_coords)
                for page, fe, img_bytes, bbox_coords in batch
            ])
            logger.info(
                "Figures described: %d/%d (batch %d-%d)",
                completed, total,
                batch_start + 1, min(batch_start + BATCH, total),
            )

    async def _recognize_table(self, img_bytes: bytes) -> str | None:
        """Recognize a table image via Qwen-VL and return markdown table format."""
        try:
            from src.generation.llm_client import LLMClient

            llm = LLMClient(provider="qwen")
            prompt = (
                "请将此图片中的表格转换为 Markdown 表格格式。"
                "只输出表格的 Markdown，不要输出其他内容。"
                "如果图片中不包含表格，回复 NOT_A_TABLE。"
            )
            result = await llm.chat_with_image(
                prompt,
                img_bytes,
                system="你是一个表格结构识别专家。将表格图片转换为 Markdown 表格。",
            )
            if result and "NOT_A_TABLE" not in result.upper():
                return result.strip()
        except Exception:
            logger.exception("Table recognition via Qwen-VL failed")
        return None

    async def _recover_tables(self, document: Document, file_path: str) -> None:
        """Recover table structures for all table layout elements via Qwen-VL.

        For PDF pages: crops the rendered page image at the table bbox,
        sends to Qwen-VL, and appends the resulting Markdown as a new Block.

        For Word docs: table blocks already have text content extracted by
        WordLoader — this method is a no-op for those.
        """
        if not file_path.lower().endswith(".pdf"):
            return  # Word table text already extracted

        import numpy as np

        from src.domain import Block

        # Collect every table crop first, then recognise in parallel batches —
        # Qwen-VL API calls are the bottleneck and are sequential otherwise.
        tasks: list[tuple] = []  # (page, table_el, img_bytes, x0, y0, x1, y1)
        for page in document.pages:
            # Find table layout elements on this page
            table_elements = [
                le for le in page.layout_elements
                if le.category.lower() in ("table", "table_caption", "table_content")
            ]
            if not table_elements:
                continue

            # Render the page once
            page_img = self._page_to_array(page, file_path, scale=1.5)
            if page_img is None:
                continue

            page_h, page_w = page_img.shape[:2]
            scale_x = page_w / page.width
            scale_y = page_h / page.height

            for table_el in table_elements:
                try:
                    # Crop table from page image
                    x0, y0, x1, y1 = _unpack_bbox(table_el.bbox)
                    px0 = max(0, int(x0 * scale_x))
                    py0 = max(0, int(y0 * scale_y))
                    px1 = min(page_w, int(x1 * scale_x))
                    py1 = min(page_h, int(y1 * scale_y))

                    if px1 <= px0 or py1 <= py0:
                        continue

                    crop = page_img[py0:py1, px0:px1]
                    pil_img = Image.fromarray(crop.astype(np.uint8))
                    tasks.append((page, table_el, _encode_vl_image(pil_img), x0, y0, x1, y1))
                except Exception:
                    logger.exception("Table crop failed: page %d", page.page_num)

        if not tasks:
            return

        BATCH = 5  # concurrent Qwen-VL calls (429 handled by retry)
        for batch_start in range(0, len(tasks), BATCH):
            batch = tasks[batch_start:batch_start + BATCH]
            markdowns = await asyncio.gather(
                *(self._recognize_table(t[2]) for t in batch),
            )
            for (page, table_el, _, x0, y0, x1, y1), markdown in zip(batch, markdowns):
                if not markdown:
                    continue
                # Append as a table block
                page.blocks.append(Block(
                    content=markdown,
                    block_type="table",
                    page_num=page.page_num,
                    bbox=(x0, y0, x1, y1),
                    reading_order=table_el.reading_order or 0,
                    confidence=table_el.confidence,
                    metadata={"source": "qwen_vl_table_recovery"},
                ))
                page.text += markdown + "\n"
                logger.info(
                    "Table recovered: page %d bbox (%.0f,%.0f,%.0f,%.0f) → %d chars",
                    page.page_num, x0, y0, x1, y1, len(markdown),
                )
            logger.info(
                "Tables recovered: %d/%d (batch %d-%d)",
                min(batch_start + BATCH, len(tasks)), len(tasks),
                batch_start + 1, min(batch_start + BATCH, len(tasks)),
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
