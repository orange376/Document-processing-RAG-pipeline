"""FastAPI application factory for the RAG pipeline API."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.config import get_settings
from src.utils import setup_logging

from .auth import verify_token, get_limiter
from .routers import admin, query, review, upload

logger = logging.getLogger(__name__)

# New Vue SPA build (frontend/dist) — falls back to the legacy static page.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
_LEGACY_STATIC = Path(__file__).resolve().parent / "static"

# Configure structured logging once at module import
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: optionally warm inference models at startup.

    The embed + reranker models load lazily on first use. Pre-warming them at
    startup makes the first query fast (~+3-5s on the first call instead) but
    keeps ~1-2GB RAM + 1.6GB GPU resident. Off by default — set
    ``WARMUP_MODELS=true`` only if you query frequently and can spare the memory.
    """
    warmup_task = None
    if get_settings().warmup_models:
        warmup_task = asyncio.create_task(asyncio.to_thread(query.warmup))
    try:
        yield
    finally:
        if warmup_task is not None:
            warmup_task.cancel()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured FastAPI instance with all routers registered.
    """
    app = FastAPI(
        title="RAG Pipeline API",
        description="Document processing and retrieval-augmented generation API",
        version="0.2.0",
        dependencies=[Depends(verify_token)],  # global auth
        lifespan=lifespan,
    )

    # ---- Rate limiter ----
    limiter = get_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ---- Routers ----
    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(review.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")

    # ---- Serve the SPA frontend (Vue build), with history-mode fallback ----
    spa_dir = _FRONTEND_DIST if _FRONTEND_DIST.is_dir() else None
    if spa_dir is not None:
        assets_dir = spa_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str) -> HTMLResponse:
            # API routes are matched first; anything else is SPA routing.
            html = (spa_dir / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(html)
    elif _LEGACY_STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(_LEGACY_STATIC)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> HTMLResponse:
            html = (_LEGACY_STATIC / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(html)

    return app
