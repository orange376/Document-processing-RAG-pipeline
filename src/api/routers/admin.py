"""Admin and health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["admin"])


@router.get("/health", summary="Health check")
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "rag-pipeline"}


@router.get("/system/queue", summary="Document processing queue status")
async def queue_status() -> dict:
    """Return real-time processing queue state.

    ``running`` lists task_ids currently being processed (at most
    ``max_concurrent``), ``waiting`` lists task_ids in FIFO order still
    awaiting a free slot. Useful for the frontend to render "排队中第 N 位".
    """
    from src.api.routers.upload import MAX_CONCURRENT, _get_processing_queue_or_none

    queue = _get_processing_queue_or_none()
    if queue is None:
        # Queue not created yet (lazy per-loop init) — report the configured cap.
        return {
            "max_concurrent": MAX_CONCURRENT,
            "running": [],
            "running_count": 0,
            "waiting": [],
            "waiting_count": 0,
        }
    return {
        "max_concurrent": queue.max_concurrent,
        "running": queue.running,
        "running_count": queue.running_count,
        "waiting": queue.waiting,
        "waiting_count": queue.waiting_count,
    }
