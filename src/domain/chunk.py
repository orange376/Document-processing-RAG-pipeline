from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .enums import ChunkType


@dataclass
class ChunkMetadata:
    """每个切片的强制元数据 — 用于溯源引用"""
    source_file: str
    page_num: int
    section: str
    chunk_type: str
    bbox: tuple[float, float, float, float] | None = None
    layout_tree_path: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """结构感知切片 — RAG 的最小检索单位"""
    chunk_id: str = field(default_factory=lambda: f"chk_{uuid4().hex[:12]}")
    content: str = ""
    metadata: ChunkMetadata | None = None
    embedding: list[float] | None = None

    def to_context_block(self) -> str:
        """生成带元数据的上下文块（用于 LLM Prompt 注入）"""
        meta = self.metadata
        if not meta:
            return self.content

        header = (
            f"[来源: {meta.source_file} | "
            f"第{meta.page_num}页"
        )
        if meta.section:
            header += f" | §{meta.section}"
        header += f" | 类型: {meta.chunk_type}]"
        return f"{header}\n{self.content}\n{header}"


@dataclass
class SearchResult:
    """语义检索结果 — 带分数和方法来源"""
    chunk: Chunk
    score: float
    retrieval_method: str = "hybrid"


@dataclass
class CitationSource:
    """引用溯源 — 从检索结果到最终答案的引用"""
    source_file: str
    page_num: int
    section: str
    chunk_type: str
    text: str


@dataclass
class QueryResult:
    """RAG 查询结果 — 答案附带引用和置信度"""
    answer: str
    citations: list[CitationSource]
    confidence: float = 0.0
    confidence_details: dict[str, float] = field(default_factory=dict)
    needs_review: bool = False
