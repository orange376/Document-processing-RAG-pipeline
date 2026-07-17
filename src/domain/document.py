from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from .enums import ProcessingStatus


@dataclass
class Block:
    """文档中的最小内容块"""
    block_id: str = field(default_factory=lambda: f"blk_{uuid4().hex[:12]}")
    content: str = ""
    block_type: str = "text"
    page_num: int = 0
    bbox: tuple[float, float, float, float] | None = None
    reading_order: int = 0
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Page:
    """文档的单个页面"""
    page_num: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)
    layout_elements: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    images: list[bytes] = field(default_factory=list)
    text: str = ""
    raw_dict: dict | None = None
    """PyMuPDF page.get_text('dict') 原始输出，供启发式版面分析使用"""


@dataclass
class Document:
    """文档的完整表示 — 解析管线的最终输出"""
    doc_id: str = field(default_factory=lambda: f"doc_{uuid4().hex[:12]}")
    filename: str = ""
    file_path: str = ""
    file_type: str = ""  # "pdf" | "docx"
    total_pages: int = 0
    pages: list[Page] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.UPLOADED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: str | None = None
