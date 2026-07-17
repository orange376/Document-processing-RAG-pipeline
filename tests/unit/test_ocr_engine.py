import pytest
from src.parser.ocr import OCREngine


class TestOCREngine:
    def test_initialization(self):
        engine = OCREngine()
        assert engine is not None
        assert engine._reader is None  # lazy load

    def test_recognize_dummy(self):
        import numpy as np

        engine = OCREngine()
        dummy = np.zeros((800, 800, 3), dtype=np.uint8)
        result = engine.recognize(dummy)
        # easyocr 在空白图上可能返回空字符串或乱码，但不应该崩溃
        assert isinstance(result, str)

    def test_unload(self):
        engine = OCREngine()
        engine.unload()
        assert True
