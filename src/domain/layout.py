from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BBox:
    """边界框 — 所有空间定位的基础"""
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BBox) -> float:
        """计算与另一个 BBox 的 IoU"""
        x_left = max(self.x0, other.x0)
        y_top = max(self.y0, other.y0)
        x_right = min(self.x1, other.x1)
        y_bottom = min(self.y1, other.y1)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class LayoutElement:
    """版面分析输出的单个元素"""
    bbox: BBox
    category: str
    confidence: float
    reading_order: int
    text: str = ""
