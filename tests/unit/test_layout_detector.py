import pytest
from src.parser.layout import LayoutDetector


class TestLayoutDetector:
    def test_initialization(self):
        detector = LayoutDetector()
        assert detector is not None

    def test_analyze_blocks_empty(self):
        detector = LayoutDetector()
        elements = detector.analyze_from_blocks([], page_num=1, page_width=595, page_height=842)
        assert elements == []

    def test_analyze_blocks_single_text(self):
        """模拟 PyMuPDF dict 格式的单文本块"""
        detector = LayoutDetector()
        blocks = [
            {
                "type": 0,
                "bbox": (72, 72, 500, 100),
                "lines": [
                    {
                        "bbox": (72, 72, 500, 100),
                        "spans": [
                            {"text": "第一章 引言", "size": 18, "font": "helv"}
                        ],
                    }
                ],
            },
            {
                "type": 0,
                "bbox": (72, 120, 500, 200),
                "lines": [
                    {
                        "bbox": (72, 120, 500, 200),
                        "spans": [
                            {"text": "这是正文内容段落。", "size": 11, "font": "helv"}
                        ],
                    }
                ],
            },
        ]
        elements = detector.analyze_from_blocks(blocks, page_num=1, page_width=595, page_height=842)
        assert len(elements) == 2
        assert elements[0].category in ("title", "section_heading")
        assert elements[1].category == "text"

    def test_analyze_blocks_image(self):
        """图片块应该被检测为 figure"""
        detector = LayoutDetector()
        blocks = [
            {"type": 1, "bbox": (72, 72, 500, 300)},
        ]
        elements = detector.analyze_from_blocks(blocks, page_num=1, page_width=595, page_height=842)
        assert len(elements) == 1
        assert elements[0].category == "figure"

    def test_detect_two_column(self):
        """双栏布局检测"""
        detector = LayoutDetector()
        # 模拟双栏：左栏 x1=240（<= mid_x - gap=249.9），右栏 x0=340（>= mid_x + gap=345.1... hmm）
        # 调整数据让 gap 能容纳
        blocks = [
            {"type": 0, "bbox": (50, 100, 240, 200)},
            {"type": 0, "bbox": (50, 210, 240, 300)},
            {"type": 0, "bbox": (50, 310, 240, 400)},
            {"type": 0, "bbox": (350, 100, 550, 200)},
            {"type": 0, "bbox": (350, 210, 550, 300)},
        ]
        is_two = detector._detect_two_column(blocks, page_width=595)
        assert is_two is True

    def test_detect_single_column(self):
        """单栏不应该被误判为双栏"""
        detector = LayoutDetector()
        blocks = [
            {"type": 0, "bbox": (72, 72, 500, 150)},
            {"type": 0, "bbox": (72, 160, 500, 250)},
        ]
        is_two = detector._detect_two_column(blocks, page_width=595)
        assert is_two is False

    def test_classify_title(self):
        detector = LayoutDetector()
        assert detector._classify_block(18, "标题") == "title"
        assert detector._classify_block(24, "大标题") == "title"

    def test_classify_heading(self):
        detector = LayoutDetector()
        assert detector._classify_block(14, "1.1 背景") == "section_heading"
        assert detector._classify_block(12, "子节标题") == "section_heading"

    def test_classify_table_caption(self):
        detector = LayoutDetector()
        assert detector._classify_block(10, "Table 2: Results") == "table_caption"
        assert detector._classify_block(10, "table 2: results") == "table_caption"
        assert detector._classify_block(10, "表 1：实验数据") == "table_caption"

    def test_unload(self):
        detector = LayoutDetector()
        detector.unload()
        assert True

    def test_analyze_image_fallback(self):
        """老的 analyze(image) 接口返回空列表"""
        detector = LayoutDetector()
        result = detector.analyze(None)
        assert result == []
