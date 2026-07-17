"""Request models for the RAG pipeline API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Query request to the RAG pipeline."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    document_ids: list[str] | None = None
