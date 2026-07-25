from __future__ import annotations

import logging

from src.config import get_settings
from src.domain.chunk import CitationSource, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting — try tiktoken first; fall back to char-based estimate.
# ---------------------------------------------------------------------------
_TOKENIZER: object | None = None  # tiktoken.Encoding


def _get_tokenizer() -> object | None:
    """Return a cached tiktoken tokenizer, or None if unavailable."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        import tiktoken

        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER = False  # sentinel — don't retry
    return _TOKENIZER if _TOKENIZER is not False else None


def _count_tokens(text: str) -> int:
    """Count tokens in *text* using tiktoken, or estimate via char count.

    For Chinese text the char:token ratio is roughly 1.5:1 on average with
    cl100k_base, while English is roughly 4:1.  We use a conservative 2:1
    estimate as the fallback.
    """
    enc = _get_tokenizer()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text) // 2  # conservative char→token estimate


# ---------------------------------------------------------------------------
# Default limits
# ---------------------------------------------------------------------------
_DEFAULT_MAX_CONTEXT_TOKENS: int = 3600
"""Default max context tokens — leaves headroom for system prompt + answer
within typical 4096-token model limits."""


class ContextBuilder:
    """将检索结果组装为带引用的上下文字符串，支持 token 窗口管理。

    检索结果按 reranker 得分排序后，从高分到低分累计加入上下文，
    直到达到 token 上限；超出部分被截断，保证 LLM 输入不超限。
    """

    def __init__(self, max_context_tokens: int | None = None):
        if max_context_tokens is None:
            s = get_settings()
            max_context_tokens = getattr(s, "max_context_tokens", None) or _DEFAULT_MAX_CONTEXT_TOKENS
        self._max_tokens = max_context_tokens

    def build(
        self,
        results: list[SearchResult],
        max_context_tokens: int | None = None,
    ) -> tuple[str, list[CitationSource]]:
        """组装上下文并提取引用源，按 token 上限截断。

        Args:
            results: 检索结果列表（应已被 reranker 排序）。
            max_context_tokens: 覆盖构造时设置的 token 上限。

        Returns:
            (上下文字符串, 引用源列表)。
        """
        if not results:
            return "", []

        max_tokens = max_context_tokens or self._max_tokens

        blocks: list[str] = []
        sources: list[CitationSource] = []
        header_template = "[{idx}] [来源: {src} | 第{page}页"

        running_tokens = 0
        truncated = False

        for i, r in enumerate(results):
            chunk = r.chunk
            block_text = chunk.to_context_block()
            block_tokens = _count_tokens(block_text + "\n\n")

            if running_tokens + block_tokens > max_tokens:
                truncated = True
                # If this is the first block and it alone exceeds the limit,
                # truncate the block content itself rather than omitting it.
                if i == 0:
                    block_text = _truncate_text_to_tokens(block_text, max_tokens)
                    running_tokens = max_tokens
                else:
                    break

            if running_tokens > 0:
                running_tokens += _count_tokens("\n\n")

            blocks.append(f"[{i + 1}] {block_text}")
            running_tokens += block_tokens

            # Build citation
            meta = chunk.metadata
            if meta:
                sources.append(
                    CitationSource(
                        source_file=meta.source_file,
                        page_num=meta.page_num,
                        section=meta.section,
                        chunk_type=meta.chunk_type,
                        text=chunk.content,
                    )
                )

        if truncated:
            logger.info(
                "Context truncated: %d/%d results included (%d tokens/%d limit)",
                len(blocks), len(results), running_tokens, max_tokens,
            )

        return "\n\n".join(blocks), sources


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to approximately *max_tokens* at a sentence boundary."""
    max_chars = max_tokens * 2  # conservative estimate
    if len(text) <= max_chars:
        return text

    # Try to find a sentence boundary
    truncated = text[:max_chars]
    for boundary in ["。", "！", "？", "\n", ".", "!", "?"]:
        pos = truncated.rfind(boundary)
        if pos > max_chars // 2:
            return truncated[:pos + 1] + "\n[内容过长，已截断]"

    return truncated + "\n[内容过长，已截断]"
