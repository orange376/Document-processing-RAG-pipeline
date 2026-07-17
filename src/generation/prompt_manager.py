class PromptManager:
    """Simple prompt template manager using str.format()."""

    TEMPLATES: dict[str, str] = {
        "qa": """你是一个专业的知识库问答助手。请基于提供的上下文回答问题。

上下文：
{context}

问题：{query}

要求：
1. 只基于上下文提供的信息回答
2. 如果上下文不足以回答问题，明确说"根据提供的资料无法回答"
3. 保持客观，不要添加自己的知识
4. 在回答中引用具体来源，使用 [来源: 文件名 | 第N页] 格式

回答：""",
        "rewrite": """将以下用户问题改写成更适合检索的形式：
原问题：{query}
改写：""",
    }

    def render(self, template_name: str, **kwargs: object) -> str:
        """Render a prompt template by name, substituting the given keyword arguments.

        Args:
            template_name: Name of the template (e.g. "qa", "rewrite").
            **kwargs: Variables to substitute into the template.

        Returns:
            The rendered prompt string.

        Raises:
            ValueError: If the template name is not recognised.
        """
        template = self.TEMPLATES.get(template_name)
        if template is None:
            raise ValueError(f"Unknown template: {template_name}")
        return template.format(**kwargs)
