"""Structured logging with loguru — replaces standard logging with rich,
coloured console output + JSON file rotation for production observability.

Usage::

    # At app startup (once):
    setup_logging()

    # In any module:
    from src.utils import timed, get_logger
    logger = get_logger(__name__)

    @timed("ocr_recognition")
    def heavy_function():
        ...

Uses loguru so there is **no** ``Logger`` class to instantiate —
every module calls ``get_logger(__name__)`` which returns a bound logger
proxy.  Standard-library ``logging`` calls are intercepted automatically.
"""

from __future__ import annotations

import logging as _stdlib_logging
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from loguru import logger as _loguru_logger


# ---------------------------------------------------------------------------
# Intercept standard logging → loguru
# ---------------------------------------------------------------------------

class _InterceptHandler(_stdlib_logging.Handler):
    """Forward stdlib log records to loguru."""

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        # Get corresponding loguru level
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the calling frame
        frame, depth = _stdlib_logging.currentframe(), 2
        while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str = "./data/logs",
    level: str = "INFO",
    json_rotation: str = "10 MB",
    json_retention: str = "7 days",
) -> None:
    """Configure loguru as the application-wide logger.

    Args:
        log_dir: Directory for JSON log files.
        level: Minimum log level for stderr.
        json_rotation: When to rotate JSON log files.
        json_retention: How long to keep JSON log files.
    """
    # Remove default loguru handler
    _loguru_logger.remove()

    # Console: colourised for development
    _loguru_logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # JSON file: structured for production / grep
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _loguru_logger.add(
        str(Path(log_dir) / "rag_{time:YYYY-MM-DD}.jsonl"),
        format="{time} | {level} | {name}:{function}:{line} | {message} | {extra}",
        level="DEBUG",
        rotation=json_rotation,
        retention=json_retention,
        compression="gz",
        serialize=True,  # JSON Lines
    )

    # Intercept standard library logging
    _stdlib_logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Silence overly verbose libraries
    for noisy in (
        "httpx", "httpcore", "urllib3", "watchfiles",
        "qdrant_client", "sentence_transformers",
    ):
        _stdlib_logging.getLogger(noisy).setLevel(_stdlib_logging.WARNING)

    _loguru_logger.info("Logging configured — level=%s, log_dir=%s", level, log_dir)


def get_logger(name: str) -> Any:
    """Return a loguru logger bound to *name*.

    Returns the global loguru logger with ``{name}`` bound in extra,
    so log records from the module carry its identity.
    """
    return _loguru_logger.bind(name=name)


# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------

def timed(label: str | None = None) -> Callable:
    """Decorator: log wall-clock time of the wrapped function.

    Usage::

        @timed("embedding")
        def embed_chunks(chunks):
            ...

    The label defaults to ``module.func_name`` when omitted.

    """

    def decorator(func: Callable) -> Callable:
        _label = label or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                _loguru_logger.debug(
                    "[timing] %s completed in %.3fs", _label, elapsed
                )
                return result
            except Exception:
                elapsed = time.perf_counter() - t0
                _loguru_logger.warning(
                    "[timing] %s failed after %.3fs", _label, elapsed
                )
                raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                _loguru_logger.debug(
                    "[timing] %s completed in %.3fs", _label, elapsed
                )
                return result
            except Exception:
                elapsed = time.perf_counter() - t0
                _loguru_logger.warning(
                    "[timing] %s failed after %.3fs", _label, elapsed
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
