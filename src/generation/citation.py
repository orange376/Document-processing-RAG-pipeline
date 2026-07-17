from __future__ import annotations

from src.domain.chunk import CitationSource


def format_citation(source: CitationSource) -> str:
    """将 CitationSource 格式化为人类可读的溯源引用字符串。

    Args:
        source: 引用溯源对象。

    Returns:
        格式化的引用字符串，例如:
        "📄 doc.pdf | 📍 第3页 | 📂 §Results"
    """
    parts = [f"📄 {source.source_file}"]
    parts.append(f"📍 第{source.page_num}页")
    if source.section:
        parts.append(f"📂 §{source.section}")
    return " | ".join(parts)
