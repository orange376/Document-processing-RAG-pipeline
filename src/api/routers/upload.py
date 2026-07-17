"""Upload and status endpoints for document processing."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from src.api.schemas import UploadResponse, ParseResponse, ParsePage, ParseBlock, ParseChunk
from src.config import get_settings
from src.domain.enums import ProcessingStatus
from src.index.shared import bm25_index
from src.pipeline import PipelineOrchestrator

router = APIRouter(tags=["documents"])

# ---------------------------------------------------------------------------
# In-memory task store
# ---------------------------------------------------------------------------
task_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------
def _make_task_entry(
    task_id: str,
    status: str,
    filename: str = "",
    file_path: str = "",
    error: str | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "status": status,
        "filename": filename,
        "file_path": file_path,
        "error": error or "",
    }


async def _process_document(task_id: str, file_path: str) -> None:
    """Process a document in the background and update the task store."""
    try:
        orchestrator = PipelineOrchestrator()
        result = await orchestrator.process_document(file_path)

        # Index chunks into BM25 for keyword retrieval
        if result.chunks:
            bm25_index.add_documents(result.chunks)

        # Simple heuristic: if no chunks were indexed, flag for review
        needs_review = result.indexed_count == 0 and result.status != ProcessingStatus.FAILED

        status = "review" if needs_review else result.status.value

        task_store[task_id] = _make_task_entry(
            task_id=task_id,
            status=status,
            filename=task_store[task_id].get("filename", ""),
            file_path=file_path,
        )
        task_store[task_id]["document_id"] = result.document.doc_id
        task_store[task_id]["indexed_count"] = result.indexed_count
        task_store[task_id]["chunk_count"] = len(result.chunks)
        task_store[task_id]["needs_review"] = needs_review

    except Exception as exc:
        task_store[task_id] = _make_task_entry(
            task_id=task_id,
            status="failed",
            filename=task_store[task_id].get("filename", ""),
            file_path=file_path,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    summary="Upload a document for processing",
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF or DOCX document and start asynchronous processing.

    Returns a ``task_id`` that can be used to poll processing status.
    """
    task_id = str(uuid.uuid4())
    settings = get_settings()

    upload_dir = settings.resolved_upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file to disk
    safe_name = f"{task_id}_{file.filename}"
    file_path = str(upload_dir / safe_name)

    content = await file.read()
    Path(file_path).write_bytes(content)

    # Initial task entry
    task_store[task_id] = _make_task_entry(
        task_id=task_id,
        status="queued",
        filename=file.filename or "unknown",
        file_path=file_path,
    )

    # Kick off background processing
    asyncio.create_task(_process_document(task_id, file_path))

    return UploadResponse(
        task_id=task_id,
        status="queued",
        message="Document queued for processing.",
    )


@router.get(
    "/documents/{task_id}/status",
    summary="Get document processing status",
)
async def get_document_status(task_id: str) -> UploadResponse:
    """Return the current processing status of an upload task."""
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return UploadResponse(
        task_id=task_id,
        status=task["status"],
        message=task.get("error", ""),
    )


# ---------------------------------------------------------------------------
# Pure parse endpoint (no indexing, no RAG)
# ---------------------------------------------------------------------------


@router.post(
    "/documents/parse",
    response_model=ParseResponse,
    summary="Parse a document and return its structure (no indexing)",
)
async def parse_document(file: UploadFile = File(...)) -> ParseResponse:
    """Upload a document and parse it, returning the full structure.

    Runs: load → layout analysis → OCR → layout tree → chunking.
    Does NOT index into vector store or BM25 — pure structural parse.

    Useful as a standalone document parser consumable by other agents.
    """
    task_id = str(uuid.uuid4())
    settings = get_settings()

    upload_dir = settings.resolved_upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"parse_{task_id}_{file.filename}"
    file_path = str(upload_dir / safe_name)

    content = await file.read()
    Path(file_path).write_bytes(content)

    # Run parse synchronously (typically fast for most documents)
    orchestrator = PipelineOrchestrator()
    try:
        result = await orchestrator.parse_document(file_path)
    except Exception as exc:
        return ParseResponse(
            task_id=task_id,
            filename=file.filename or "unknown",
            status="failed",
            message=str(exc),
        )
    finally:
        # Clean up the temp file
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass

    doc = result.document

    pages_out = []
    for page in doc.pages:
        blocks_out = [
            ParseBlock(
                content=b.content,
                block_type=b.block_type,
                page_num=b.page_num,
                bbox=b.bbox,
                reading_order=b.reading_order,
            )
            for b in page.blocks
        ]
        pages_out.append(ParsePage(
            page_num=page.page_num,
            width=page.width,
            height=page.height,
            blocks=blocks_out,
            text_length=len(page.text),
        ))

    chunks_out = []
    for chunk in result.chunks:
        meta = chunk.metadata
        if meta:
            chunks_out.append(ParseChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source_file=meta.source_file,
                page_num=meta.page_num,
                section=meta.section,
                chunk_type=meta.chunk_type,
                bbox=meta.bbox,
            ))
        else:
            chunks_out.append(ParseChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
            ))

    return ParseResponse(
        task_id=task_id,
        filename=doc.filename,
        file_type=doc.file_type,
        total_pages=doc.total_pages,
        pages=pages_out,
        chunks=chunks_out,
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        message=f"Parsed {doc.total_pages} page(s), {len(result.chunks)} chunk(s).",
    )
