from enum import Enum, auto


class BlockType(str, Enum):
    """版面元素类型 — 对应 PP-DocLayoutV3 的 26 类输出"""
    TITLE = "title"
    SECTION_HEADING = "section_heading"
    TEXT = "text"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    FORMULA = "formula"
    FORMULA_CAPTION = "formula_caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"
    STAMP = "stamp"
    OTHER = "other"


class ChunkType(str, Enum):
    """切片类型"""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CODE = "code"
    MIXED = "mixed"


class ProcessingStatus(str, Enum):
    """文档处理状态"""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXING = "indexing"
    SCORING = "scoring"
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"
    INDEXED = "indexed"
    FAILED = "failed"
