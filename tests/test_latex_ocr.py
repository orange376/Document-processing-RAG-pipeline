"""LatexOCREngine unit tests."""

import io

import pytest

pix2tex = pytest.importorskip("pix2tex", reason="pix2tex not installed")

from src.ocr import LatexOCREngine  # noqa: E402


class TestLatexOCREngine:
    def test_recognize_returns_tuple(self):
        """recognize 总是返回 (str, float)，即便识别失败。"""
        from PIL import Image

        engine = LatexOCREngine()
        buf = io.BytesIO()
        Image.new("RGB", (64, 32), "white").save(buf, format="PNG")
        latex, conf = engine.recognize(buf.getvalue())
        assert isinstance(latex, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_recognize_wraps_in_dollars(self):
        """返回的 LaTeX 用 $$ 包裹（若模型输出非空）。"""
        from PIL import Image

        engine = LatexOCREngine()
        buf = io.BytesIO()
        Image.new("RGB", (128, 64), "white").save(buf, format="PNG")
        latex, _ = engine.recognize(buf.getvalue())
        # 空结果或 $$ 包裹均可（识别失败返回 ""，不强制非空）
        assert latex == "" or latex.startswith("$$")
