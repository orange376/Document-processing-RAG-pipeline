from __future__ import annotations

from src.domain.chunk import CitationSource, SearchResult


class ContextBuilder:
    """将检索结果组装为带引用的上下文字符串。"""

    def build(self, results: list[SearchResult]) -> tuple[str, list[CitationSource]]:
        """组装上下文并提取引用源。

        Args:
            results: 检索结果列表。

        Returns:
            (上下文字符串, 引用源列表)。
        """
        if not results:
            return "", []

        blocks: list[str] = []
        sources: list[CitationSource] = []

        for i, r in enumerate(results):
            chunk = r.chunk
            block = chunk.to_context_block()
            blocks.append(f"[{i + 1}] {block}")

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

        return "\n\n".join(blocks), sources
