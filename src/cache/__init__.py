"""Redis-backed cache layer for expensive compute results.

- Embedding vectors (by content hash)
- Reranker scores (by query + chunk_id)
- LLM answers (by query + context hash)

All cache keys are namespaced: ``rag:embed:v2:{sha256}`` etc.
TTL defaults are tuned for each cache tier.
"""

from .redis_cache import RedisCache

__all__ = ["RedisCache"]
