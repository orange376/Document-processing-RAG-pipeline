"""Response models for the RAG pipeline API."""

from pydantic import BaseModel, Field

from src.domain.chunk import CitationSource


class UploadResponse(BaseModel):
    """Response for a document upload request."""

    task_id: str
    status: str = "processing"
    message: str = ""


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


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    status_code: int = 500
