"""Authentication and rate-limiting middleware for the RAG pipeline API.

Authentication
--------------
If ``API_AUTH_TOKEN`` is set in the environment, every non-health-check request
must include a matching ``Authorization: Bearer <token>`` header.  When the
variable is unset, auth is **disabled** (open mode — suitable for local dev).

Rate Limiting
-------------
Uses slowapi (a FastAPI-compatible rate limiter).  Default limits:

- Upload: 10 requests per minute per client
- Query:  20 requests per minute per client
- Review: 30 requests per minute per client
- Health: unlimited

All limits are configurable via settings / env vars.
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
_limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


def get_limiter() -> Limiter:
    return _limiter


# ---------------------------------------------------------------------------
# Token authentication
# ---------------------------------------------------------------------------

def _get_auth_token() -> str:
    """Return the expected API auth token, or empty string to disable auth."""
    s = get_settings()
    return getattr(s, "api_auth_token", "") or ""


async def verify_token(request: Request) -> None:
    """FastAPI dependency: verify Bearer token on every request.

    Skips health-check, static, and OPTIONS requests.
    When ``API_AUTH_TOKEN`` is not configured, all requests pass through.
    """
    # Skip auth for health checks and static files
    path = request.url.path
    if path.endswith("/health") or path.startswith("/static") or request.method == "OPTIONS":
        return

    expected = _get_auth_token()
    if not expected:
        return  # auth disabled

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:]  # strip "Bearer "
    if token != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Convenience: combined dependency
# ---------------------------------------------------------------------------

AuthDep = Depends(verify_token)
"""Drop-in FastAPI dependency — use ``auth: None = AuthDep`` on route params."""
