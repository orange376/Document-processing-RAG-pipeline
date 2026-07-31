"""本地公式识别引擎 — 基于 pix2tex (LaTeX-OCR)。

将公式图片转换为 LaTeX，本地 CPU 推理，不依赖外部 API。
权重首次使用时从 HuggingFace 下载（~200MB），之后离线可用。
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

_MODEL: object | None = None


def _get_model() -> object:
    """返回共享的 pix2tex LatexOCR 单例。"""
    global _MODEL
    if _MODEL is None:
        from pix2tex.cli import LatexOCR

        _MODEL = LatexOCR(no_cuda=True)
    return _MODEL


class LatexOCREngine:
    """基于 pix2tex 的本地公式识别引擎。

    用法::

        engine = LatexOCREngine()
        latex, confidence = engine.recognize(image_bytes)
    """

    def recognize(self, image_bytes: bytes) -> tuple[str, float]:
        """识别公式图片为 LaTeX。

        Returns:
            (latex_string, confidence)。
            latex 用 ``$$...$$`` 包裹；识别失败返回 ("", 0.0)。
        """
        try:
            from PIL import Image

            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            model = _get_model()
            latex = model(pil_img)
            latex = (latex or "").strip()
            if not latex:
                return "", 0.0
            if not latex.startswith("$$"):
                latex = f"$${latex}$$"
            return latex, 0.85
        except Exception:
            logger.warning("pix2tex formula recognition failed", exc_info=True)
            return "", 0.0
