from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import get_settings
from src.domain import Page, Block

if TYPE_CHECKING:
    import numpy as np


class OCREngine:
    """OCR 引擎 — 基于 easyocr（纯 torch，支持 GPU）

    支持的语种：ch_sim（简体中文）, en（英文）, 等。
    首次运行时自动下载模型权重到 ~/.EasyOCR/model/。
    """

    def __init__(self, languages: list[str] | None = None):
        self._reader = None
        self._languages = languages or ["ch_sim", "en"]
        self._settings = get_settings()

    def _lazy_load(self):
        if self._reader is None:
            import easyocr
            # gpu=True 会自动使用 CUDA（如果 torch 检测到 GPU）
            self._reader = easyocr.Reader(
                self._languages,
                gpu=True,  # torch.cuda.is_available() 自动判断
                model_storage_directory=None,  # 默认 ~/.EasyOCR/model/
            )

    def unload(self):
        """卸载模型释放 VRAM"""
        self._reader = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def recognize(self, image: np.ndarray, page: Page | None = None) -> Page | str:
        """OCR 识别单页图像

        Args:
            image: numpy array (H, W, 3) RGB
            page: 可选的 Page 对象，若提供则更新其 blocks

        Returns:
            若 page 为 None，返回识别文本；
            若 page 不为 None，返回更新后的 Page
        """
        self._lazy_load()

        # easyocr.readtext 返回 [(bbox, text, confidence), ...]
        results = self._reader.readtext(image, detail=1, paragraph=True)

        lines = []
        for bbox, text, confidence in results:
            lines.append(text)
            if page is not None:
                # bbox 格式: [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                block = Block(
                    content=text,
                    block_type="text",
                    page_num=page.page_num,
                    bbox=(min(x_coords), min(y_coords), max(x_coords), max(y_coords)),
                    confidence=float(confidence),
                )
                page.blocks.append(block)

        recognized_text = "\n".join(lines)

        if page is not None:
            page.text = recognized_text
            return page

        return recognized_text


def create_ocr_engine() -> OCREngine:
    return OCREngine()
