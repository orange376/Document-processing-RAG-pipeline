"""Response models for the RAG pipeline API."""

from pydantic import BaseModel, Field

from src.domain.chunk import CitationSource


class UploadResponse(BaseModel):
    """Response for a document upload request."""

    task_id: str
    status: str = "processing"
    message: str = ""
    filename: str = ""
    indexed_count: int = 0
    chunk_count: int = 0
    total_pages: int = 0
    needs_review: bool = False


class ParseBlock(BaseModel):
    """A single parsed block within a page."""

    content: str = ""
    block_type: str = "text"
    page_num: int = 0
    bbox: tuple[float, float, float, float] | None = None
    reading_order: int = 0


class ParsePage(BaseModel):
    """A single parsed page."""

    page_num: int
    width: float
    height: float
    blocks: list[ParseBlock] = []
    text_length: int = 0


class ParseChunk(BaseModel):
    """A chunk produced by structure-aware chunking."""

    chunk_id: str
    content: str = ""
    source_file: str = ""
    page_num: int = 0
    section: str = ""
    chunk_type: str = "text"
    bbox: tuple[float, float, float, float] | None = None


class ParseResponse(BaseModel):
    """Response for a pure document parse request."""

    task_id: str
    filename: str = ""
    file_type: str = ""
    total_pages: int = 0
    pages: list[ParsePage] = []
    chunks: list[ParseChunk] = []
    status: str = "ok"
    message: str = ""


class QueryResponse(BaseModel):
    """Response for a RAG query."""

    answer: str
    citations: list[CitationSource]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_details: dict[str, float] = Field(default_factory=dict)
    needs_review: bool = False


class ReviewBlock(BaseModel):
    """A block shown in the review detail."""

    block_id: str
    content: str
    block_type: str = "text"
    page_num: int = 0
    confidence: float = 0.0
    editable: bool = True


class ReviewChunk(BaseModel):
    """A chunk shown in the review detail."""

    chunk_id: str
    content: str = ""
    source_file: str = ""
    page_num: int = 0
    section: str = ""
    chunk_type: str = "text"
    layout_tree_path: list[str] = Field(default_factory=list)


class ReviewPage(BaseModel):
    """A page shown in the review detail."""

    page_num: int
    blocks: list[ReviewBlock] = []
    text_length: int = 0


class ReviewDetail(BaseModel):
    """Full review detail for a single task."""

    task_id: str
    filename: str = ""
    file_type: str = ""
    status: str = ""
    total_pages: int = 0
    total_chunks: int = 0
    indexed_count: int = 0
    confidence: float = 0.0
    confidence_details: dict[str, float] = Field(default_factory=dict)
    pages: list[ReviewPage] = []
    chunks: list[ReviewChunk] = []
    error: str = ""
    needs_review: bool = False


class ReviewAction(BaseModel):
    """Action to take on a review task."""

    action: str = Field(default="approve", pattern="^(approve|reject)$")
    reason: str = ""
    edited_blocks: list[dict] = Field(default_factory=list)
    """List of {block_id, new_content} dicts to apply before approving."""


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    status_code: int = 500
