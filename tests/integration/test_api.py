"""Integration tests for the FastAPI application using TestClient.

These tests mock out heavy dependencies (retriever, embedding engine, LLM client,
orchestrator) so they can run without a GPU, Qdrant, or remote API keys.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.schemas import UploadResponse
from src.domain import Chunk, ChunkMetadata, CitationSource, ProcessingStatus
from src.domain.chunk import SearchResult
from src.pipeline import ProcessingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create the FastAPI application."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a TestClient for the application."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Upload endpoint tests
# ---------------------------------------------------------------------------


class TestUploadDocument:
    """Tests for POST /api/v1/documents/upload and GET /api/v1/documents/{task_id}/status."""

    @patch("src.api.routers.upload.PipelineOrchestrator")
    def test_upload_and_status_check(self, mock_orch_cls, client):
        """Upload a file and poll its processing status."""
        # Mock orchestrator to return a successful result immediately
        mock_result = MagicMock()
        mock_result.document = MagicMock()
        mock_result.document.doc_id = "doc_mock"
        mock_result.indexed_count = 5
        mock_result.chunks = [MagicMock() for _ in range(3)]
        mock_result.status = ProcessingStatus.INDEXED

        mock_instance = AsyncMock()
        mock_instance.process_document.return_value = mock_result
        mock_orch_cls.return_value = mock_instance

        # Upload a dummy PDF
        fake_file = io.BytesIO(b"%PDF-1.4 mock content")
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", fake_file, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        task_id = data["task_id"]

        # Poll status — the background task should have completed quickly
        import time
        time.sleep(0.2)

        status_resp = client.get(f"/api/v1/documents/{task_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["task_id"] == task_id

    def test_upload_no_file_returns_422(self, client):
        """Uploading without a file should return 422."""
        response = client.post("/api/v1/documents/upload")
        assert response.status_code == 422

    def test_status_not_found(self, client):
        """Querying a non-existent task should return 404."""
        response = client.get("/api/v1/documents/nonexistent/status")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Parse endpoint tests
# ---------------------------------------------------------------------------


class TestParseDocument:
    """Tests for POST /api/v1/documents/parse."""

    @patch("src.api.routers.upload.PipelineOrchestrator")
    def test_parse_success_pdf(self, mock_orch_cls, client):
        """Parsing a valid PDF returns pages and chunks."""
        # Build a realistic mock result
        from src.domain import Block, Page, Document, ProcessingStatus

        doc = Document(
            filename="report.pdf",
            file_type="pdf",
            total_pages=2,
        )
        doc.pages = [
            Page(
                page_num=1, width=595, height=842,
                blocks=[
                    Block(content="Introduction", block_type="title", page_num=1,
                          bbox=(50, 50, 200, 80), reading_order=0),
                    Block(content="This is the intro text.", block_type="text", page_num=1,
                          bbox=(50, 100, 500, 150), reading_order=1),
                ],
                text="Introduction\nThis is the intro text.\n",
            ),
            Page(
                page_num=2, width=595, height=842,
                blocks=[
                    Block(content="Data", block_type="section_heading", page_num=2,
                          bbox=(50, 50, 150, 80), reading_order=0),
                    Block(content="Some data content.", block_type="text", page_num=2,
                          bbox=(50, 100, 400, 130), reading_order=1),
                ],
                text="Data\nSome data content.\n",
            ),
        ]

        from src.domain.chunk import Chunk as DomainChunk, ChunkMetadata
        chunks = [
            DomainChunk(
                chunk_id="chk_1", content="Introduction\nThis is the intro text.",
                metadata=ChunkMetadata(source_file="report.pdf", page_num=1,
                                       section="Introduction", chunk_type="text"),
            ),
            DomainChunk(
                chunk_id="chk_2", content="Data\nSome data content.",
                metadata=ChunkMetadata(source_file="report.pdf", page_num=2,
                                       section="Data", chunk_type="text"),
            ),
        ]

        mock_result = MagicMock()
        mock_result.document = doc
        mock_result.chunks = chunks
        mock_result.status = ProcessingStatus.PROCESSING

        mock_instance = AsyncMock()
        mock_instance.parse_document.return_value = mock_result
        mock_orch_cls.return_value = mock_instance

        # Send a dummy PDF
        fake_file = io.BytesIO(b"%PDF-1.4 mock content")
        response = client.post(
            "/api/v1/documents/parse",
            files={"file": ("report.pdf", fake_file, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "report.pdf"
        assert data["file_type"] == "pdf"
        assert data["total_pages"] == 2
        assert len(data["pages"]) == 2
        assert len(data["chunks"]) == 2
        assert data["status"] == "processing"

        # Check first page structure
        first_page = data["pages"][0]
        assert first_page["page_num"] == 1
        assert len(first_page["blocks"]) == 2
        assert first_page["blocks"][0]["content"] == "Introduction"
        assert first_page["blocks"][0]["block_type"] == "title"

        # Check first chunk metadata
        first_chunk = data["chunks"][0]
        assert first_chunk["source_file"] == "report.pdf"
        assert first_chunk["page_num"] == 1
        assert first_chunk["section"] == "Introduction"

    @patch("src.api.routers.upload.PipelineOrchestrator")
    def test_parse_word_document(self, mock_orch_cls, client):
        """Parsing a .docx returns pages and chunks with heading structure."""
        from src.domain import Block, Page, Document, ProcessingStatus
        from src.domain.chunk import Chunk as DomainChunk, ChunkMetadata

        # Simulate WordLoader output: single page, headings, paragraphs, table
        doc = Document(filename="report.docx", file_type="docx", total_pages=1)
        doc.pages = [
            Page(page_num=1, width=595, height=842, blocks=[
                Block(content="Introduction", block_type="title", page_num=1,
                      bbox=(0, 0, 595, 20), reading_order=0),
                Block(content="This is the intro.", block_type="text", page_num=1,
                      bbox=(0, 0, 595, 20), reading_order=1),
                Block(content="1.1 Background", block_type="section_heading", page_num=1,
                      bbox=(0, 0, 595, 20), reading_order=2),
                Block(content="Background details.", block_type="text", page_num=1,
                      bbox=(0, 0, 595, 20), reading_order=3),
                Block(content="Item | Count\nThing | 5", block_type="table", page_num=1,
                      bbox=(0, 0, 595, 40), reading_order=4),
            ], text="Introduction\nThis is the intro.\n1.1 Background\nBackground details.\nItem | Count\nThing | 5\n"),
        ]

        chunks = [
            DomainChunk(chunk_id="chk_w1", content="Introduction\nThis is the intro.",
                        metadata=ChunkMetadata(source_file="report.docx", page_num=1,
                                               section="Introduction", chunk_type="text")),
            DomainChunk(chunk_id="chk_w2", content="1.1 Background\nBackground details.",
                        metadata=ChunkMetadata(source_file="report.docx", page_num=1,
                                               section="1.1 Background", chunk_type="text")),
            DomainChunk(chunk_id="chk_w3", content="Item | Count\nThing | 5",
                        metadata=ChunkMetadata(source_file="report.docx", page_num=1,
                                               section="1.1 Background", chunk_type="table")),
        ]

        mock_result = MagicMock()
        mock_result.document = doc
        mock_result.chunks = chunks
        mock_result.status = ProcessingStatus.PROCESSING

        mock_instance = AsyncMock()
        mock_instance.parse_document.return_value = mock_result
        mock_orch_cls.return_value = mock_instance

        fake_file = io.BytesIO(b"PK\x03\x04 mock docx content")
        response = client.post(
            "/api/v1/documents/parse",
            files={"file": ("report.docx", fake_file,
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "report.docx"
        assert data["file_type"] == "docx"
        assert data["total_pages"] >= 1
        assert len(data["pages"]) == 1
        assert data["pages"][0]["page_num"] == 1

        # Check headings in blocks
        blocks = data["pages"][0]["blocks"]
        titles = [b for b in blocks if b["block_type"] == "title"]
        assert any("Introduction" in b["content"] for b in titles)
        sections = [b for b in blocks if b["block_type"] == "section_heading"]
        assert any("Background" in b["content"] for b in sections)
        tables = [b for b in blocks if b["block_type"] == "table"]
        assert len(tables) == 1

        # Check chunks
        assert len(data["chunks"]) == 3

    @patch("src.api.routers.upload.PipelineOrchestrator")
    def test_parse_unsupported_format_falls_through(self, mock_orch_cls, client):
        """Uploading an unsupported file type should still be attempted (orchestrator raises)."""
        mock_instance = AsyncMock()
        mock_instance.parse_document.side_effect = ValueError("不支持的文件格式")
        mock_orch_cls.return_value = mock_instance

        fake_file = io.BytesIO(b"not a real document")
        response = client.post(
            "/api/v1/documents/parse",
            files={"file": ("notes.txt", fake_file, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"

    def test_parse_no_file_returns_422(self, client):
        """Uploading without a file should return 422."""
        response = client.post("/api/v1/documents/parse")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Query endpoint tests
# ---------------------------------------------------------------------------


class TestQuery:
    """Tests for POST /api/v1/query."""

    @pytest.fixture(autouse=True)
    def _mock_retriever(self):
        """Mock the Retriever to return canned results."""
        chunk = Chunk(
            chunk_id="chk_test",
            content="RAG pipelines combine retrieval with generation.",
            metadata=ChunkMetadata(
                source_file="test_doc.pdf",
                page_num=1,
                section="1.0",
                chunk_type="text",
            ),
        )
        mock_result = SearchResult(chunk=chunk, score=0.92, retrieval_method="hybrid")

        patcher = patch(
            "src.api.routers.query._build_retriever",
            return_value=MagicMock(retrieve=MagicMock(return_value=[mock_result])),
        )
        patcher.start()
        yield
        patcher.stop()

    @pytest.fixture(autouse=True)
    def _mock_embedding(self):
        """Mock the EmbeddingEngine to return a dummy vector."""
        mock_engine = MagicMock()
        mock_engine.embed.return_value = [0.1] * 1024
        mock_engine.unload.return_value = None

        patcher = patch(
            "src.api.routers.query._build_embedding_engine",
            return_value=mock_engine,
        )
        patcher.start()
        yield
        patcher.stop()

    @pytest.fixture(autouse=True)
    def _mock_llm(self):
        """Mock the LLMClient to return a canned answer."""

        def chat_side_effect(prompt, system="", temperature=0.3, max_tokens=2048):
            if "根据提供的资料无法回答" in prompt:
                return "根据提供的资料无法回答"
            return "RAG pipelines combine retrieval with generation for grounded answers."

        mock_llm = MagicMock()
        mock_llm.chat.side_effect = chat_side_effect

        patcher = patch(
            "src.api.routers.query._build_llm_client",
            return_value=mock_llm,
        )
        patcher.start()
        yield
        patcher.stop()

    @pytest.fixture(autouse=True)
    def _mock_confidence(self):
        """Mock the ConfidenceScorer to return a fixed high score."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {
            "overall": 0.85,
            "details": {"reranker_score": 0.92},
        }

        patcher = patch(
            "src.api.routers.query._build_confidence_scorer",
            return_value=mock_scorer,
        )
        patcher.start()
        yield
        patcher.stop()

    def test_query_success(self, client):
        """A normal RAG query returns an answer with citations."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What is a RAG pipeline?", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "citations" in data
        assert len(data["citations"]) > 0
        assert data["confidence"] >= 0.0
        assert isinstance(data["confidence_details"], dict)
        # High enough confidence -> no review needed
        assert data["needs_review"] is False

    def test_query_empty_query_returns_422(self, client):
        """An empty query string should fail validation."""
        response = client.post("/api/v1/query", json={"query": ""})
        assert response.status_code == 422

    def test_query_low_confidence(self, client):
        """When confidence is low, the fallback answer should be used."""
        # Re-mock the scorer to return very low confidence
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {
            "overall": 0.1,
            "details": {"reranker_score": 0.05},
        }

        with patch(
            "src.api.routers.query._build_confidence_scorer",
            return_value=mock_scorer,
        ):
            response = client.post(
                "/api/v1/query",
                json={"query": "obscure topic", "top_k": 3},
            )

        assert response.status_code == 200
        data = response.json()
        assert "置信度较低" in data["answer"]
        assert len(data["citations"]) == 0
        assert data["needs_review"] is False

    @patch("src.api.routers.query._build_retriever")
    def test_query_no_results(self, mock_builder, client):
        """When retrieval returns nothing, a no-answer message is returned."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_builder.return_value = mock_retriever

        response = client.post(
            "/api/v1/query",
            json={"query": "very obscure topic", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "未找到相关文档" in data["answer"]
        assert data["citations"] == []
        assert data["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Review endpoint tests
# ---------------------------------------------------------------------------


class TestReview:
    """Tests for review endpoints."""

    def test_list_pending_empty(self, client):
        """With no pending reviews, the list is empty."""
        response = client.get("/api/v1/review/pending")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_pending_with_data(self, client):
        """Add a task flagged for review and verify it appears in the pending list."""
        from src.api.routers.upload import task_store

        # Manually insert a task that needs review
        task_store["review_test_1"] = {
            "task_id": "review_test_1",
            "status": "review",
            "filename": "test.pdf",
            "file_path": "/tmp/test.pdf",
            "needs_review": True,
            "indexed_count": 0,
            "chunk_count": 0,
        }

        response = client.get("/api/v1/review/pending")
        assert response.status_code == 200
        items = response.json()
        assert len(items) >= 1
        task_ids = [item["task_id"] for item in items]
        assert "review_test_1" in task_ids

        # Cleanup
        del task_store["review_test_1"]

    def test_approve_review(self, client):
        """Approving a review task changes its status."""
        from src.api.routers.upload import task_store

        task_store["approve_test"] = {
            "task_id": "approve_test",
            "status": "review",
            "filename": "doc.pdf",
            "file_path": "/tmp/doc.pdf",
            "needs_review": True,
            "indexed_count": 0,
            "chunk_count": 0,
        }

        response = client.post(
            "/api/v1/review/approve_test/approve",
            json={"action": "approve"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["decision"] == "approve"

        # Cleanup
        del task_store["approve_test"]

    def test_reject_review(self, client):
        """Rejecting a review task changes its status."""
        from src.api.routers.upload import task_store

        task_store["reject_test"] = {
            "task_id": "reject_test",
            "status": "review",
            "filename": "doc.pdf",
            "file_path": "/tmp/doc.pdf",
            "needs_review": True,
            "indexed_count": 0,
            "chunk_count": 0,
        }

        response = client.post(
            "/api/v1/review/reject_test/approve",
            json={"action": "reject"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["decision"] == "reject"

        # Cleanup
        del task_store["reject_test"]

    def test_approve_nonexistent_task(self, client):
        """Approving a non-existent task returns 404."""
        response = client.post(
            "/api/v1/review/nonexistent/approve",
            json={"action": "approve"},
        )
        assert response.status_code == 404

    def test_approve_task_not_needing_review(self, client):
        """Approving a task that doesn't need review returns 400."""
        from src.api.routers.upload import task_store

        task_store["no_review"] = {
            "task_id": "no_review",
            "status": "completed",
            "filename": "doc.pdf",
            "file_path": "/tmp/doc.pdf",
            "needs_review": False,
            "indexed_count": 5,
            "chunk_count": 3,
        }

        response = client.post(
            "/api/v1/review/no_review/approve",
            json={"action": "approve"},
        )
        assert response.status_code == 400

        # Cleanup
        del task_store["no_review"]


# ---------------------------------------------------------------------------
# Admin endpoint tests
# ---------------------------------------------------------------------------


class TestAdmin:
    """Tests for admin endpoints."""

    def test_health_check(self, client):
        """Health check returns ok."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "rag-pipeline"


# ---------------------------------------------------------------------------
# App factory test
# ---------------------------------------------------------------------------


class TestAppFactory:
    """Tests for the create_app factory."""

    def test_create_app_returns_app(self):
        """create_app returns a valid FastAPI instance."""
        app = create_app()
        assert app.title == "RAG Pipeline API"

    def test_app_has_routes(self):
        """The app should have the expected routes registered."""
        app = create_app()
        routes = set()
        for route in app.routes:
            # Standard routes (openapi, docs, etc.)
            if hasattr(route, "path") and not hasattr(route, "original_router"):
                routes.add(route.path)
            # Included routers — prefix + inner route path
            if hasattr(route, "original_router") and hasattr(route, "include_context"):
                prefix = route.include_context.prefix.rstrip("/")
                for r in route.original_router.routes:
                    if hasattr(r, "path"):
                        routes.add(f"{prefix}{r.path}")
        assert "/api/v1/documents/upload" in routes
        assert "/api/v1/documents/parse" in routes
        assert "/api/v1/documents/{task_id}/status" in routes
        assert "/api/v1/query" in routes
        assert "/api/v1/review/pending" in routes
        assert "/api/v1/review/{task_id}/approve" in routes
        assert "/api/v1/health" in routes
