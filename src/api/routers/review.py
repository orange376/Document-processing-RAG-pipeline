"""Review endpoints for document processing tasks."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.schemas import ErrorResponse
from src.api.routers.upload import task_store

router = APIRouter(tags=["review"])


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ReviewAction(BaseModel):
    """Action to take on a review task."""

    action: str = Field(default="approve", pattern="^(approve|reject)$")


class ReviewItem(BaseModel):
    """A single pending review item."""

    task_id: str
    filename: str
    status: str
    indexed_count: int = 0
    chunk_count: int = 0


# ---------------------------------------------------------------------------
# In-memory review store (can persist decisions)
# ---------------------------------------------------------------------------
review_decisions: dict[str, str] = {}  # task_id -> "approved" | "rejected"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/review/pending",
    response_model=list[ReviewItem],
    summary="List all tasks pending review",
)
async def list_pending_reviews() -> list[ReviewItem]:
    """Return all document processing tasks that need human review."""
    pending: list[ReviewItem] = []
    for task_id, task in task_store.items():
        if task.get("needs_review") and task_id not in review_decisions:
            pending.append(
                ReviewItem(
                    task_id=task_id,
                    filename=task.get("filename", ""),
                    status=task.get("status", ""),
                    indexed_count=task.get("indexed_count", 0),
                    chunk_count=task.get("chunk_count", 0),
                )
            )
    return pending


@router.post(
    "/review/{task_id}/approve",
    summary="Approve or reject a pending review task",
)
async def approve_review(task_id: str, body: ReviewAction) -> dict[str, str]:
    """Approve or reject a document processing task that was flagged for review.

    - ``approve``: marks the task as accepted.
    - ``reject``: marks the task as rejected.
    """
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = task_store[task_id]
    if not task.get("needs_review"):
        raise HTTPException(status_code=400, detail=f"Task {task_id} does not need review")

    decision = body.action
    if decision == "approve":
        task["status"] = "accepted"
        review_decisions[task_id] = "approved"
    else:
        task["status"] = "rejected"
        review_decisions[task_id] = "rejected"

    return {"task_id": task_id, "status": task["status"], "decision": decision}
