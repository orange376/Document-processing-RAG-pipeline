import pytest
from src.domain import LayoutElement, BBox
from src.parser.layout_tree import LayoutTreeBuilder, LayoutTreeNode


@pytest.fixture
def sample_elements():
    return [
        LayoutElement(bbox=BBox(0, 0, 100, 20, 1), category="title",
                      confidence=0.95, text="第一章 引言", reading_order=0),
        LayoutElement(bbox=BBox(0, 25, 100, 50, 1), category="text",
                      confidence=0.90, text="这是引言段落。", reading_order=1),
        LayoutElement(bbox=BBox(0, 55, 100, 75, 1), category="title",
                      confidence=0.95, text="1.1 背景", reading_order=2),
        LayoutElement(bbox=BBox(0, 80, 100, 100, 1), category="text",
                      confidence=0.90, text="背景介绍内容。", reading_order=3),
        LayoutElement(bbox=BBox(0, 105, 100, 130, 1), category="table",
                      confidence=0.85, text="TABLE_DATA", reading_order=4),
    ]


class TestLayoutTreeBuilder:
    def test_build_with_headings(self, sample_elements):
        builder = LayoutTreeBuilder()
        tree = builder.build(sample_elements)

        assert tree.category == "root"
        assert len(tree.children) == 2  # 两个标题

        # 第一个标题是"第一章 引言"
        assert tree.children[0].category == "title"
        assert "引言" in tree.children[0].text

        # 它下面有一个 text 子节点
        assert len(tree.children[0].children) == 1

    def test_build_no_headings(self):
        elements = [
            LayoutElement(bbox=BBox(0, 0, 100, 30, 1), category="text",
                          confidence=0.9, text="段落1", reading_order=0),
            LayoutElement(bbox=BBox(0, 35, 100, 65, 1), category="text",
                          confidence=0.9, text="段落2", reading_order=1),
        ]
        builder = LayoutTreeBuilder()
        tree = builder.build(elements)
        assert len(tree.children) == 2  # 全部挂 root

    def test_build_empty(self):
        builder = LayoutTreeBuilder()
        tree = builder.build([])
        assert tree.category == "root"
        assert len(tree.children) == 0

    def test_to_path(self):
        node = LayoutTreeNode(category="text", bbox=None, text="test")
        assert node.to_path() == ["text"]
