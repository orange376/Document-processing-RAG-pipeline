from __future__ import annotations

from dataclasses import dataclass, field
from src.domain import LayoutElement, BBox


@dataclass
class LayoutTreeNode:
    """版面树节点"""
    category: str
    bbox: BBox | None
    text: str = ""
    children: list[LayoutTreeNode] = field(default_factory=list)
    confidence: float = 0.0

    def to_text(self, indent: int = 0) -> str:
        """递归生成带缩进的文本表示"""
        prefix = "  " * indent
        lines = [f"{prefix}[{self.category}] {self.text[:60]}"]
        for child in self.children:
            lines.append(child.to_text(indent + 1))
        return "\n".join(lines)

    def to_path(self) -> list[str]:
        """返回从根到当前节点的 path 标签列表"""
        return [self.category]


class LayoutTreeBuilder:
    """版面树构建器 — 将 LayoutElement 列表组织为层次树"""

    TITLE_KEYS = {"title", "section_heading", "heading", "h1", "h2", "h3"}

    def build(self, elements: list[LayoutElement]) -> LayoutTreeNode:
        """将 LayoutElement 列表构建为版面树

        策略：
        1. 标题节点作为分支节点
        2. 正文/表格/图片作为叶子挂到最近标题下
        3. 无标题则全部挂到 root
        """
        root = LayoutTreeNode(category="root", bbox=None, text="文档")

        if not elements:
            return root

        # 分离标题和非标题
        headings = [e for e in elements if e.category.lower() in self.TITLE_KEYS]
        others = [e for e in elements if e.category.lower() not in self.TITLE_KEYS]

        if not headings:
            # 无标题结构，全部直接挂 root
            for el in others:
                root.children.append(LayoutTreeNode(
                    category=el.category,
                    bbox=el.bbox,
                    text=el.text[:120],
                    confidence=el.confidence,
                ))
            return root

        # 有标题：每个标题下的内容归入其子树
        current_heading = root
        for el in elements:
            is_heading = el.category.lower() in self.TITLE_KEYS
            node = LayoutTreeNode(
                category=el.category,
                bbox=el.bbox,
                text=el.text[:120],
                confidence=el.confidence,
            )
            if is_heading:
                root.children.append(node)
                current_heading = node
            else:
                current_heading.children.append(node)

        return root
