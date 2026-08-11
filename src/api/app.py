"""FastAPI application factory for the RAG pipeline API."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.utils import setup_logging

from .auth import verify_token, get_limiter
from .routers import admin, query, review, upload

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Configure structured logging once at module import
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: warm inference models in the background at startup.

    The embed + reranker models load lazily on first use, which costs ~33s on
    the first query after a restart. Loading them in the background at startup
    moves that cost off the critical path without blocking API startup.
    """
    warmup_task = asyncio.create_task(asyncio.to_thread(query.warmup))
    try:
        yield
    finally:
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

    # ---- Serve static frontend ----
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> HTMLResponse:
            html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(html)

    return app
