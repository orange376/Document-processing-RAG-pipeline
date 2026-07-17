"""Tests for the API schema models."""

import pytest
from pydantic import ValidationError

from src.api.schemas import ErrorResponse, QueryRequest, QueryResponse, UploadResponse
from src.domain import CitationSource


class TestQueryRequest:
    def test_defaults(self):
        req = QueryRequest(query="test query")
        assert req.query == "test query"
        assert req.top_k == 10
        assert req.document_ids is None

    def test_custom_values(self):
        req = QueryRequest(
            query="test query",
            top_k=25,
            document_ids=["doc1", "doc2"],
        )
        assert req.query == "test query"
        assert req.top_k == 25
        assert req.document_ids == ["doc1", "doc2"]

    def test_query_empty_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_top_k_too_small_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)

    def test_top_k_too_large_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=51)


class TestUploadResponse:
    def test_defaults(self):
        resp = UploadResponse(task_id="abc123")
        assert resp.task_id == "abc123"
        assert resp.status == "processing"
        assert resp.message == ""

    def test_custom_values(self):
        resp = UploadResponse(
            task_id="abc123",
            status="completed",
            message="Upload successful.",
        )
        assert resp.task_id == "abc123"
        assert resp.status == "completed"
        assert resp.message == "Upload successful."


class TestQueryResponse:
    @pytest.fixture
    def sample_citations(self):
        return [
            CitationSource(
                source_file="doc1.pdf",
                page_num=3,
                section="2.1",
                chunk_type="text",
                text="Some content.",
            ),
        ]

    def test_defaults(self, sample_citations):
        resp = QueryResponse(answer="Test answer.", citations=sample_citations)
        assert resp.answer == "Test answer."
        assert resp.citations == sample_citations
        assert resp.confidence == 0.0
        assert resp.confidence_details == {}
        assert resp.needs_review is False

    def test_custom_values(self, sample_citations):
        resp = QueryResponse(
            answer="Test answer.",
            citations=sample_citations,
            confidence=0.85,
            confidence_details={"semantic": 0.9, "citation": 0.8},
            needs_review=True,
        )
        assert resp.confidence == 0.85
        assert resp.confidence_details == {"semantic": 0.9, "citation": 0.8}
        assert resp.needs_review is True

    def test_confidence_below_zero_raises(self, sample_citations):
        with pytest.raises(ValidationError):
            QueryResponse(
                answer="test",
                citations=sample_citations,
                confidence=-0.1,
            )

    def test_confidence_above_one_raises(self, sample_citations):
        with pytest.raises(ValidationError):
            QueryResponse(
                answer="test",
                citations=sample_citations,
                confidence=1.1,
            )


class TestErrorResponse:
    def test_defaults(self):
        resp = ErrorResponse(detail="Not found")
        assert resp.detail == "Not found"
        assert resp.status_code == 500

    def test_custom_status_code(self):
        resp = ErrorResponse(detail="Bad request", status_code=400)
        assert resp.detail == "Bad request"
        assert resp.status_code == 400
