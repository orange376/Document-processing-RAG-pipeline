"""Admin and health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["admin"])


@router.get("/health", summary="Health check")
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "rag-pipeline"}
