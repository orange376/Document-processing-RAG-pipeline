"""Layout detector using PP-DocLayoutV3 from PaddleX with heuristic fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from src.config import get_settings
from src.domain import BBox, LayoutElement

if TYPE_CHECKING:
    import numpy as np

# PP-DocLayoutV3 label mapping (from inference.yml label_list)
PP_LABEL_MAP = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "display_formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer",
    9: "footer_image",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header_image",
    14: "image",
    15: "inline_formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "vertical_text",
    24: "vision_footnote",
}

# Map PP-DocLayoutV3 labels to pipeline LayoutElement categories
PP_TO_PIPELINE_CATEGORY = {
    "doc_title": "title",
    "paragraph_title": "section_heading",
    "text": "text",
    "abstract": "text",
    "algorithm": "text",
    "aside_text": "text",
    "content": "text",
    "vertical_text": "text",
    "number": "text",
    "image": "figure",
    "chart": "figure",
    "seal": "figure",
    "figure_title": "figure_caption",
    "table": "table",
    "header": "header",
    "header_image": "header",
    "footer": "footer",
    "footer_image": "footer",
    "footnote": "footer",
    "vision_footnote": "footer",
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula_number": "formula",
    "reference": "reference",
    "reference_content": "reference",
}


class LayoutDetector:
    """Layout analyzer using PP-DocLayoutV3 from PaddleX.

    Uses deep learning model for accurate layout analysis (25 classes).
    Falls back to heuristic analysis (font size / position based) if the
    model is unavailable or fails.
    """

    def __init__(self):
        self._settings = get_settings()
        self._model = None

    # ── Model lifecycle ──

    def _lazy_load(self):
        """Lazy-load PP-DocLayoutV3 model on first use."""
        if self._model is not None:
            return
        # CRITICAL: import torch BEFORE paddle to avoid DLL conflicts
        # on Windows (paddle pollutes DLL search path causing torch shm.dll
        # to fail loading with WinError 127)
        import torch  # noqa: F401

        # Apply monkey-patch for paddle 2.x -> 3.x compatibility
        self._apply_paddle_patches()
        from paddlex import create_model

        self._model = create_model("PP-DocLayoutV3")

    @staticmethod
    def _apply_paddle_patches():
        """Apply compatibility patches for missing paddle APIs."""
        import paddle.base.libpaddle

        AnalysisConfig = paddle.base.libpaddle.AnalysisConfig
        if not hasattr(AnalysisConfig, "set_optimization_level"):
            setattr(
                AnalysisConfig,
                "set_optimization_level",
                AnalysisConfig.set_tensorrt_optimization_level,
            )

    def unload(self):
        """Release model resources."""
        self._model = None
        import gc

        gc.collect()

    # ── Image-based analysis (PP-DocLayoutV3) ──

    def analyze(
        self,
        image: np.ndarray | str,
        scale: float = 1.0,
    ) -> list[LayoutElement]:
        """Run PP-DocLayoutV3 on a page image.

        Args:
            image: Numpy array (H, W, C) or file path string.
            scale: Factor to divide coordinates by (e.g. 2.0 for 2x render).

        Returns:
            List of LayoutElement sorted by reading order.
        """
        try:
            self._lazy_load()
        except Exception:
            # PP-DocLayoutV3 not available (paddle not installed, etc.)
            return []
        try:
            results = self._model.predict(image, batch_size=1)
            elements: list[LayoutElement] = []
            for res in results:
                page_elements = self._parse_result(res, scale)
                elements.extend(page_elements)

            # Sort by reading order (top-to-bottom, left-to-right)
            elements.sort(key=lambda e: (e.bbox.y0, e.bbox.x0))
            for i, el in enumerate(elements):
                el.reading_order = i
            return elements
        except Exception:
            # Graceful degradation
            return []

    def _parse_result(
        self, result: Any, scale: float
    ) -> list[LayoutElement]:
        """Parse PP-DocLayoutV3 prediction result into LayoutElements."""
        elements: list[LayoutElement] = []
        boxes = result.get("boxes", [])

        for box in boxes:
            cls_id = box.get("cls_id", -1)
            label = box.get("label", "text")
            score = box.get("score", 0.5)
            coord = box.get("coordinate", [0, 0, 0, 0])

            if cls_id < 0 or len(coord) < 4:
                continue

            # Scale coordinates back from render space to page points
            x0 = coord[0] / scale
            y0 = coord[1] / scale
            x1 = coord[2] / scale
            y1 = coord[3] / scale

            category = PP_TO_PIPELINE_CATEGORY.get(label, "text")

            elements.append(
                LayoutElement(
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    category=category,
                    confidence=score,
                    reading_order=0,
                    text="",
                )
            )

        return elements

    # ── Block-based heuristic analysis (fallback) ──

    def analyze_from_blocks(
        self,
        page_blocks: list[dict],
        page_num: int,
        page_width: float,
        page_height: float,
    ) -> list[LayoutElement]:
        """Heuristic layout analysis from PyMuPDF dict blocks.

        Used as fallback when PP-DocLayoutV3 is not available or for
        text-only PDFs where model-based analysis is unnecessary.
        """
        elements: list[LayoutElement] = []

        if not page_blocks:
            return elements

        is_two_column = self._detect_two_column(page_blocks, page_width)
        mid_x = page_width / 2.0
        column_gap = page_width * 0.08

        left_col: list[LayoutElement] = []
        right_col: list[LayoutElement] = []

        for block in page_blocks:
            block_type = block.get("type", 0)
            bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

            # --- Image block ---
            if block_type == 1:
                el = LayoutElement(
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1, page_num=page_num),
                    category="figure",
                    confidence=0.9,
                    reading_order=0,
                )
                left_col.append(el)
                continue

            # --- Text block ---
            max_font_size = 0.0
            full_text = ""
            has_lines = False

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s.get("text", "") for s in spans)
                if not line_text.strip():
                    continue
                has_lines = True
                full_text += line_text
                font_size = spans[0].get("size", 11)
                if font_size > max_font_size:
                    max_font_size = font_size

            if not has_lines and not full_text.strip():
                continue

            category = self._classify_block(max_font_size, full_text)

            el = LayoutElement(
                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1, page_num=page_num),
                category=category,
                confidence=0.85,
                reading_order=0,
                text=full_text.strip(),
            )

            if is_two_column and x1 < mid_x - column_gap:
                left_col.append(el)
            elif is_two_column and x0 > mid_x + column_gap:
                right_col.append(el)
            else:
                left_col.append(el)

        left_col.sort(key=lambda e: e.bbox.y0)
        right_col.sort(key=lambda e: e.bbox.y0)

        if is_two_column and right_col:
            ordered: list[LayoutElement] = []
            li, ri = 0, 0
            while li < len(left_col) and ri < len(right_col):
                if left_col[li].bbox.y0 <= right_col[ri].bbox.y0:
                    ordered.append(left_col[li])
                    li += 1
                else:
                    ordered.append(right_col[ri])
                    ri += 1
            ordered.extend(left_col[li:])
            ordered.extend(right_col[ri:])
        else:
            ordered = left_col

        for i, el in enumerate(ordered):
            el.reading_order = i

        return ordered

    @staticmethod
    def _classify_block(font_size: float, text: str) -> str:
        """Classify block category by font size and text content."""
        text_stripped = text.strip()
        low = text_stripped.lower()

        if low.startswith("table ") or low.startswith("表 "):
            return "table_caption"
        if low.startswith("figure ") or low.startswith("图 "):
            return "figure_caption"

        if font_size >= 18:
            return "title"
        if font_size >= 14:
            return "section_heading"
        if font_size >= 12:
            return "section_heading"

        return "text"

    @staticmethod
    def _detect_two_column(
        blocks: list[dict], page_width: float
    ) -> bool:
        """Detect if the page uses a two-column layout."""
        if not blocks or page_width == 0:
            return False
        mid_x = page_width / 2.0
        gap = page_width * 0.08
        left_count = 0
        right_count = 0
        for block in blocks:
            if block.get("type", 0) != 0:
                continue
            bbox = block.get("bbox", (0, 0, 0, 0))
            x0, x1 = bbox[0], bbox[2]
            if x1 <= mid_x - gap:
                left_count += 1
            elif x0 >= mid_x + gap:
                right_count += 1
        total = left_count + right_count
        if total < 3:
            return False
        return left_count / total >= 0.25 and right_count / total >= 0.25


def create_layout_detector() -> LayoutDetector:
    return LayoutDetector()
