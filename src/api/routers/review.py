"""Review endpoints — document quality inspection and human approval."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ReviewAction,
    ReviewBlock,
    ReviewChunk,
    ReviewDetail,
    ReviewPage,
)
from src.api.routers.upload import _save_task_db, task_store
from src.pipeline import PipelineOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["review"])

# In-memory review decisions (task_id -> "approved" | "rejected")
review_decisions: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_review_detail(task_id: str, task: dict) -> ReviewDetail:
    """Build a ReviewDetail from a task entry, with safe defaults."""
    # Reconstruct pages with ReviewBlock objects
    pages_raw: list[dict] = task.get("pages", [])
    pages_out = []
    for p in pages_raw:
        blocks_out = [
            ReviewBlock(
                block_id=b.get("block_id", ""),
                content=b.get("content", ""),
                block_type=b.get("block_type", "text"),
                page_num=b.get("page_num", 0),
                confidence=b.get("confidence", 0.0),
                editable=True,
            )
            for b in p.get("blocks", [])
        ]
        pages_out.append(ReviewPage(
            page_num=p.get("page_num", 0),
            blocks=blocks_out,
            text_length=p.get("text_length", 0),
        ))

    # Reconstruct chunks with ReviewChunk objects
    chunks_raw: list[dict] = task.get("chunks", [])
    chunks_out = [
        ReviewChunk(
            chunk_id=c.get("chunk_id", ""),
            content=c.get("content", ""),
            source_file=c.get("source_file", ""),
            page_num=c.get("page_num", 0),
            section=c.get("section", ""),
            chunk_type=c.get("chunk_type", "text"),
            layout_tree_path=c.get("layout_tree_path", []),
        )
        for c in chunks_raw
    ]

    return ReviewDetail(
        task_id=task_id,
        filename=task.get("filename", ""),
        file_type=task.get("file_type", ""),
        status=task.get("status", ""),
        total_pages=task.get("total_pages", 0),
        total_chunks=task.get("chunk_count", 0),
        indexed_count=task.get("indexed_count", 0),
        confidence=task.get("confidence", 0.0),
        confidence_details=task.get("confidence_details", {}),
        pages=pages_out,
        chunks=chunks_out,
        error=task.get("error", ""),
        needs_review=task.get("needs_review", False),
    )


def _apply_block_edits(task: dict, edited_blocks: list[dict]) -> None:
    """Apply block content edits in-place on the task's serialized pages."""
    pages_raw: list[dict] = task.get("pages", [])
    edit_map: dict[str, str] = {eb["block_id"]: eb["new_content"] for eb in edited_blocks if "block_id" in eb}
    if not edit_map:
        return
    for page in pages_raw:
        for block in page.get("blocks", []):
            bid = block.get("block_id", "")
            if bid in edit_map:
                block["content"] = edit_map[bid]
    task["pages"] = pages_raw


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/review/pending")
async def list_pending_reviews() -> list[ReviewDetail]:
    """Return all document processing tasks that need human review."""
    pending: list[ReviewDetail] = []
    for task_id, task in task_store.items():
        if task.get("needs_review") and task_id not in review_decisions:
            pending.append(_build_review_detail(task_id, task))
    return pending


@router.get("/review/{task_id}")
async def get_review_detail(task_id: str) -> ReviewDetail:
    """Return full review detail for a single task.

    Includes all pages, blocks (with content), chunks, and confidence scores.
    """
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return _build_review_detail(task_id, task)


@router.post("/review/{task_id}/approve")
async def approve_review(task_id: str, body: ReviewAction) -> dict:
    """Approve or reject a review task.

    - ``action``: ``"approve"`` or ``"reject"``
    - ``reason``: optional explanation (required for reject)
    - ``edited_blocks``: optional list of ``{block_id, new_content}`` to apply first
    """
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = task_store[task_id]
    if not task.get("needs_review"):
        raise HTTPException(status_code=400, detail=f"Task {task_id} does not need review")

    decision = body.action

    if decision == "reject" and not body.reason:
        raise HTTPException(status_code=400, detail="拒绝时必须填写原因")

    # Apply any block edits before changing status
    if body.edited_blocks:
        _apply_block_edits(task, body.edited_blocks)

    if decision == "approve":
        task["status"] = "accepted"
        review_decisions[task_id] = "approved"
    else:
        task["status"] = "rejected"
        task["error"] = body.reason or "用户拒绝"
        review_decisions[task_id] = "rejected"

    task["needs_review"] = False
    _save_task_db()

    return {
        "task_id": task_id,
        "status": task["status"],
        "decision": decision,
        "reason": body.reason,
    }


@router.post("/review/{task_id}/reprocess")
async def reprocess_review(task_id: str) -> dict:
    """Re-run the pipeline on the original file and update the task."""
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    file_path = task.get("file_path", "")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=400, detail=f"原始文件不存在: {file_path}")

    from src.api.routers.upload import _process_document

    # Reset status and re-process
    task["status"] = "processing"
    task["needs_review"] = False
    task["error"] = ""
    _save_task_db()

    # Re-process asynchronously
    import asyncio
    asyncio.create_task(_process_document(task_id, file_path))

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "文档已重新加入处理队列",
    }
