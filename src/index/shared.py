"""Application-scoped shared instances for heavy dependencies.

Use these singletons across modules to avoid redundant instantiation
of expensive resources (BM25 index, etc.).
"""

from __future__ import annotations

from src.index.bm25_index import BM25Index

# Singleton BM25 index that accumulates documents across uploads.
# Populated by the document upload pipeline and consumed by the retriever
# at query time.
bm25_index: BM25Index = BM25Index()
