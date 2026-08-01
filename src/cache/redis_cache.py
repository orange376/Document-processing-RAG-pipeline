"""Redis cache for embeddings, reranker scores, and LLM answers.

Connects via the ``REDIS_URL`` setting.  When Redis is unreachable, all
cache operations silently fall through — the pipeline still works, just
without caching.

Cache key scheme::

    rag:embed:v2:<sha256 of content>       →  JSON list of floats (768-dim)
    rag:rerank:<md5 of query+chunk_id>     →  float score
    rag:answer:<md5 of query+context>      →  answer string

TTL values:

- Embeddings:  7 days  (stable — same content → same vector)
- Reranker:    1 hour  (model weights are fixed, results don't vary)
- Answers:     1 hour  (LLM is deterministic-ish with same prompt)
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)

_EMBED_TTL = 7 * 24 * 3600    # 7 days
_RERANK_TTL = 3600            # 1 hour
_ANSWER_TTL = 3600            # 1 hour


# Module-level flag — only try connecting once per process lifetime
_redis_disabled: bool = False


class RedisCache:
    """Redis-backed cache for expensive RAG pipeline operations.

    When Redis is unreachable, all cache operations silently fall through
    after the first failed connection attempt — the pipeline still works,
    just without caching.

    Usage::

        cache = RedisCache()

        # Embedding
        vec = cache.get_embedding("some text content")
        if vec is None:
            vec = compute_embedding("some text content")
            cache.set_embedding("some text content", vec)
    """

    def __init__(self):
        global _redis_disabled
        self._client: object | None = None
        self._connected: bool = not _redis_disabled

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def get_embedding(self, text: str) -> list[float] | None:
        """Return the cached embedding for *text*, or None."""
        key = f"rag:embed:v2:{_sha256(text)}"
        raw = self._get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_embedding(self, text: str, vector: list[float]) -> None:
        """Cache *vector* for *text*."""
        key = f"rag:embed:v2:{_sha256(text)}"
        self._set(key, json.dumps(vector), ttl=_EMBED_TTL)

    # ------------------------------------------------------------------
    # Reranker cache
    # ------------------------------------------------------------------

    def get_rerank_score(self, query: str, chunk_id: str) -> float | None:
        """Return the cached reranker score, or None."""
        key = f"rag:rerank:{_hash_pair(query, chunk_id)}"
        raw = self._get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    def set_rerank_score(self, query: str, chunk_id: str, score: float) -> None:
        """Cache *score* for a (query, chunk_id) pair."""
        key = f"rag:rerank:{_hash_pair(query, chunk_id)}"
        self._set(key, str(score), ttl=_RERANK_TTL)

    # ------------------------------------------------------------------
    # Answer cache
    # ------------------------------------------------------------------

    def get_answer(self, query: str, context: str) -> str | None:
        """Return a cached LLM answer, or None."""
        key = f"rag:answer:{_hash_pair(query, context)}"
        return self._get(key)

    def set_answer(self, query: str, context: str, answer: str) -> None:
        """Cache an LLM *answer* for a (query, context) pair."""
        key = f"rag:answer:{_hash_pair(query, context)}"
        self._set(key, answer, ttl=_ANSWER_TTL)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get(self, key: str) -> str | None:
        self._lazy_connect()
        if not self._connected:
            return None
        try:
            return self._client.get(key)
        except Exception:
            self._connected = False
            logger.warning("Redis GET failed, cache disabled for this session")
            return None

    def _set(self, key: str, value: str, ttl: int) -> None:
        self._lazy_connect()
        if not self._connected:
            return
        try:
            self._client.setex(key, ttl, value)
        except Exception:
            self._connected = False
            logger.warning("Redis SET failed, cache disabled for this session")

    def _lazy_connect(self) -> None:
        global _redis_disabled
        if self._client is not None or _redis_disabled:
            return
        try:
            import redis

            self._client = redis.Redis.from_url(
                get_settings().redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            self._connected = True
            logger.info("Redis cache connected: %s", get_settings().redis_url)
        except Exception:
            self._client = None
            self._connected = False
            _redis_disabled = True  # don't retry for this process lifetime
            logger.info("Redis unavailable — cache disabled")


# ---------------------------------------------------------------------------
# Singleton (import-safe — connects on first use)
# ---------------------------------------------------------------------------
_cache_instance: RedisCache | None = None


def get_cache() -> RedisCache:
    """Return the module-level cache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _hash_pair(a: str, b: str) -> str:
    return hashlib.md5(f"{a}|{b}".encode("utf-8")).hexdigest()[:12]
