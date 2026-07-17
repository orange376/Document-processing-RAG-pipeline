class QueryRewriter:
    def __init__(self, llm_client=None):
        self._llm = llm_client

    def rewrite(self, query: str) -> str:
        """改写 query。LLM 不可用时返回原文。"""
        if not self._llm:
            return query
        try:
            prompt = f"将以下用户问题改写成更适合检索的形式，保持核心意图：\n{query}"
            result = self._llm.chat(prompt, system="你是一个检索优化专家。请简洁改写。")
            return result.strip() or query
        except Exception:
            return query
