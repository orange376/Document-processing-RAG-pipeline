"""本地公式识别引擎 — 基于 pix2tex (LaTeX-OCR)。

将公式图片转换为 LaTeX，本地 CPU 推理，不依赖外部 API。
权重首次使用时从 GitHub releases 下载（pix2tex 官方模型仓库），
之后离线可用。
"""

from __future__ import annotations

import io
import logging
import threading

logger = logging.getLogger(__name__)

_MODEL: object | None = None

# Serializes lazy singleton construction so concurrent asyncio.to_thread calls
# can't both build the model (check-then-act race).
_LOAD_LOCK = threading.Lock()


def _get_model() -> object:
    """返回共享的 pix2tex LatexOCR 单例。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is None:
            import torch
            from munch import Munch
            from pix2tex.cli import LatexOCR

            # pix2tex 默认 no_cuda=True（强制 CPU）。显式允许 CUDA，
            # 公式识别在 GPU 机器上快 5-10 倍。需带全默认键（pix2tex 直接
            # 访问 arguments.config / arguments.checkpoint）。
            args = Munch({
                "config": "settings/config.yaml",
                "checkpoint": "checkpoints/weights.pth",
                "no_cuda": not torch.cuda.is_available(),
                "no_resize": False,
            })
            _MODEL = LatexOCR(args)
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
            confidence 是固定占位值 0.85 —— pix2tex 不提供置信度，
            该值并非由模型计算得出。
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
