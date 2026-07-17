"""BM25 关键词索引 — 基于 rank_bm25"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

from src.domain import Chunk, SearchResult


def _tokenize(text: str) -> list[str]:
    """智能分词。

    当 jieba 可用时使用 jieba 分词（支持中文词组），
    否则回退到空格拆分（英文）与单字拆分（中文）的混合策略。
    全部转为小写以保证大小写不敏感的匹配。
    """
    try:
        import jieba

        return list(jieba.cut(text.lower()))
    except ImportError:
        tokens: list[str] = []
        for token in text.split():
            token_lower = token.lower()
            # 若 token 包含中文字符则拆为单字
            if any("一" <= c <= "鿿" for c in token_lower):
                tokens.extend(list(token_lower))
            else:
                tokens.append(token_lower)
        return tokens


_MIN_IDF: float = 0.1
"""IDF 最小值下限，防止小语料库（如单文档场景）下 BM25Okapi 产生零分。"""


class BM25Index:
    """BM25 关键词检索索引

    封装 rank_bm25.BM25Okapi，支持文档增删、检索、持久化。
    """

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._bm25: object | None = None  # rank_bm25.BM25Okapi 实例

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list[Chunk]) -> None:
        """添加文档并重建 BM25 索引（追加模式）。"""
        self._chunks.extend(chunks)
        self._rebuild()

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 30) -> list[SearchResult]:
        """BM25 检索，返回 top_k 个 SearchResult。"""
        if not query.strip() or self._bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 按分数降序排列，取 top_k
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append(
                    SearchResult(
                        chunk=self._chunks[idx],
                        score=score,
                        retrieval_method="bm25",
                    )
                )
        return results

    # ------------------------------------------------------------------
    # 清空
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """清空索引与文档列表。"""
        self._chunks.clear()
        self._bm25 = None

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """将文档列表持久化到磁盘（加载时会重建 BM25 索引）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"chunks": self._chunks}, f)

    def load(self, path: str) -> None:
        """从磁盘加载文档列表并重建 BM25 索引。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._rebuild()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """根据当前 _chunks 重建 BM25 索引。

        注：BM25Okapi 的 IDF 公式 `log(N - n + 0.5) - log(n + 0.5)`
        在词汇出现文档数恰好为语料库一半时会产生零值，这在小型语料库
        （如只有 2 份文档）中很常见。此处对 IDF 施加下限以避免零分。
        """
        from rank_bm25 import BM25Okapi

        if not self._chunks:
            self._bm25 = None
            return
        tokenized_corpus = [_tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

        # 修复零值 IDF：所有低于下限的 IDF 被提升到 _MIN_IDF
        for word, val in self._bm25.idf.items():
            if val < _MIN_IDF:
                self._bm25.idf[word] = _MIN_IDF


def create_bm25_index() -> BM25Index:
    """工厂函数（遵循项目惯例）。"""
    return BM25Index()
