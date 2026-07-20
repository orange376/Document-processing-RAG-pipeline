"""Upload and status endpoints for document processing."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from src.api.schemas import UploadResponse, ParseResponse, ParsePage, ParseBlock, ParseChunk

logger = logging.getLogger(__name__)
from src.config import get_settings
from src.domain.enums import ProcessingStatus
from src.index.shared import bm25_index
from src.index.vector_store import VectorStore
from src.pipeline import PipelineOrchestrator

router = APIRouter(tags=["documents"])

# ---------------------------------------------------------------------------
# Persistent task store (survives server restarts)
# ---------------------------------------------------------------------------
task_store: dict[str, dict] = {}

_settings_for_db = get_settings()
TASK_DB_PATH = _settings_for_db.resolved_upload_dir / "task_db.json"
BM25_PATH = _settings_for_db.resolved_upload_dir / "bm25_index.pkl"


def _save_task_db() -> None:
    """Serialize task_store to disk."""
    try:
        TASK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Filter out non-serializable entries (bytes, etc.)
        clean: dict[str, dict] = {}
        for tid, t in task_store.items():
            clean[tid] = {k: v for k, v in t.items() if _is_json_safe(v)}
        serialized = json.dumps(clean, ensure_ascii=False, indent=2, default=str)
        TASK_DB_PATH.write_text(serialized, encoding="utf-8")
    except Exception as exc:
        logger.warning("_save_task_db failed: %s", exc)


def _load_task_db() -> None:
    """Deserialize task_store from disk on startup."""
    try:
        if TASK_DB_PATH.exists():
            data = json.loads(TASK_DB_PATH.read_text(encoding="utf-8"))
            task_store.update(data)
    except Exception:
        pass


def _is_json_safe(val) -> bool:
    """Return True if *val* is trivially JSON-serializable."""
    return isinstance(val, (str, int, float, bool, list, dict, type(None)))


# Load persisted tasks on module init
_load_task_db()
try:
    if BM25_PATH.exists():
        bm25_index.load(str(BM25_PATH))
except Exception:
    pass


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


def _serialize_chunks(chunks) -> list[dict]:
    """Serialize Chunk objects to JSON-safe dicts."""
    result = []
    for c in chunks:
        item = {"chunk_id": c.chunk_id, "content": c.content, "chunk_type": "text"}
        if c.metadata:
            item["source_file"] = c.metadata.source_file
            item["page_num"] = c.metadata.page_num
            item["section"] = c.metadata.section
            item["chunk_type"] = c.metadata.chunk_type
            item["layout_tree_path"] = c.metadata.layout_tree_path
        result.append(item)
    return result


def _serialize_pages(document) -> list[dict]:
    """Serialize Page objects to JSON-safe dicts."""
    pages = []
    for p in document.pages:
        blocks = []
        for b in p.blocks:
            blocks.append({
                "block_id": b.block_id,
                "content": b.content,
                "block_type": b.block_type,
                "page_num": b.page_num,
                "confidence": b.confidence,
                "metadata": b.metadata,
            })
        pages.append({
            "page_num": p.page_num,
            "blocks": blocks,
            "text_length": len(p.text),
        })
    return pages


async def _process_document(task_id: str, file_path: str) -> None:
    """Process a document in the background and update the task store."""
    try:
        orchestrator = PipelineOrchestrator()
        result = await orchestrator.process_document(file_path)

        # Index chunks into BM25 for keyword retrieval
        if result.chunks:
            bm25_index.add_documents(result.chunks)
            bm25_index.save(str(BM25_PATH))

        # Compute multi-dimension confidence score
        from src.confidence.scorer import ConfidenceScorer
        scorer = ConfidenceScorer()

        # Gather available data for all 5 scoring dimensions
        all_layout_elements = []
        all_blocks: list = []
        for page in result.document.pages:
            all_layout_elements.extend(page.layout_elements)
            all_blocks.extend(page.blocks)

        # Tables from page.tables (domain Table objects, if populated)
        all_tables = []
        for page in result.document.pages:
            all_tables.extend(getattr(page, "tables", []))

        score_result = scorer.score(
            layout_elements=all_layout_elements,
            ocr_results=all_blocks,         # Block has .confidence attribute
            tables=all_tables,
            chunks=result.chunks,
            reranker_scores=[1.0],          # indexing time — no reranker needed
        )

        # Confidence-based review decision (replaces hard-coded chunk_coherence check)
        from src.confidence.threshold import ThresholdStrategy
        from src.config import get_settings
        _cfg = get_settings()
        threshold = ThresholdStrategy(
            accept=_cfg.confidence_threshold_accept,
            reject=_cfg.confidence_threshold_reject,
        )
        overall_conf = score_result.get("overall", 0.0)
        confidence_category = threshold.classify(overall_conf)
        needs_review = (
            confidence_category in ("review", "reject")
            or result.indexed_count == 0
        ) and result.status != ProcessingStatus.FAILED

        status = "review" if needs_review else result.status.value

        entry = _make_task_entry(
            task_id=task_id,
            status=status,
            filename=task_store[task_id].get("filename", ""),
            file_path=file_path,
        )
        entry["document_id"] = result.document.doc_id
        entry["indexed_count"] = result.indexed_count
        entry["chunk_count"] = len(result.chunks)
        entry["total_pages"] = result.document.total_pages
        entry["needs_review"] = needs_review
        entry["confidence_category"] = confidence_category
        entry["confidence"] = score_result.get("overall", 0.0)
        entry["confidence_details"] = score_result.get("details", {})
        entry["file_type"] = result.document.file_type
        entry["pages"] = _serialize_pages(result.document)
        entry["chunks"] = _serialize_chunks(result.chunks)

        task_store[task_id] = entry
        _save_task_db()

    except Exception as exc:
        task_store[task_id] = _make_task_entry(
            task_id=task_id,
            status="failed",
            filename=task_store[task_id].get("filename", ""),
            file_path=file_path,
            error=str(exc),
        )
        _save_task_db()


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
    _save_task_db()

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
        filename=task.get("filename", ""),
        indexed_count=task.get("indexed_count", 0),
        chunk_count=task.get("chunk_count", 0),
        total_pages=task.get("total_pages", 0),
        needs_review=task.get("needs_review", False),
    )


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@router.delete(
    "/documents/{task_id}",
    summary="Delete an uploaded document and its index entries",
)
async def delete_document(task_id: str) -> dict:
    """Delete a document from storage and the BM25 index.

    Removes the task entry, deletes the uploaded file from disk,
    and purges its chunks from the BM25 index.
    """
    task = task_store.pop(task_id, None)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Delete the uploaded file
    file_path = task.get("file_path", "")
    if file_path:
        try:
            p = Path(file_path)
            if p.exists():
                p.unlink()
        except Exception as exc:
            logger.warning("Failed to delete file %s: %s", file_path, exc)

    # Remove from BM25 index
    # NOTE: chunk metadata stores source_file = Path(file_path).name
    # (the prefixed on-disk name, e.g. "abc123_高数题.docx"), not the
    # original upload name.  We use the stored file_path here.
    disk_name = Path(file_path).name if file_path else ""
    if disk_name:
        try:
            removed = bm25_index.remove_by_source_file(disk_name)
            bm25_index.save(str(BM25_PATH))
            logger.info("Removed %d chunks for %s from BM25 index", removed, disk_name)
        except Exception as exc:
            logger.warning("Failed to remove from BM25 index: %s", exc)

    # Remove from Qdrant vector store
    if disk_name:
        try:
            vs = VectorStore()
            vr = vs.delete_by_source_file(disk_name)
            logger.info("Removed %d points for %s from vector store", vr, disk_name)
        except Exception as exc:
            logger.warning("Failed to remove from vector store: %s", exc)

    _save_task_db()
    return {"task_id": task_id, "status": "deleted"}


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
