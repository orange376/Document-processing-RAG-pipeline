class QueryRewriter:
    """Query 改写器 — 将用户查询改写成更适合 BM25 关键词检索的形式。

    改写策略：
    - 提取核心关键词和短语
    - 补充同义词/近义词
    - 展开缩写和简写
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client

    def rewrite(self, query: str) -> str:
        """改写 query。LLM 不可用时返回原文。"""
        if not self._llm:
            return query
        try:
            prompt = (
                f"原始查询：{query}\n\n"
                "请将上述查询改写成更适合 BM25 关键词检索的形式：\n"
                "1. 提取核心关键词和重要短语\n"
                "2. 补充常见同义词或等价表述\n"
                "3. 保留专有名词和公式符号\n"
                "4. 保持查询简洁，不要添加原始查询没有的新信息\n"
                "直接输出改写后的查询文本，不要解释。"
            )
            result = self._llm.chat(
                prompt,
                system="你是一个检索优化专家。将用户查询改写为同义关键词组合以提高检索召回率。",
            )
            return result.strip() or query
        except Exception:
            return query
