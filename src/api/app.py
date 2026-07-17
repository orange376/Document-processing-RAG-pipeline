"""FastAPI application factory for the RAG pipeline API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routers import admin, query, review, upload

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured FastAPI instance with all routers registered.
    """
    app = FastAPI(
        title="RAG Pipeline API",
        description="Document processing and retrieval-augmented generation API",
        version="0.1.0",
    )

    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(review.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")

    # Serve static frontend
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> HTMLResponse:
            html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(html)

    return app
