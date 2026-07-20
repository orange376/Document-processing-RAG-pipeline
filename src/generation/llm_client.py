from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

import httpx
from src.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM API client — supports DeepSeek (default) and Qwen providers.

    Usage::

        # DeepSeek for language generation
        llm = LLMClient()  # defaults to provider="deepseek"

        # Qwen for multimodal
        vl_llm = LLMClient(provider="qwen")
    """

    def __init__(self, api_key: str | None = None, provider: str = "deepseek"):
        s = get_settings()
        self._provider = provider

        if provider == "qwen":
            self._api_key = api_key or s.qwen_api_key
            self._base = s.qwen_api_base
            self._model = s.qwen_model
        else:
            self._api_key = api_key or s.deepseek_api_key
            self._base = s.deepseek_api_base
            self._model = s.deepseek_model

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        if not self._api_key:
            return ""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = httpx.post(
                f"{self._base}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("LLM API call failed (%s)", self._provider)
            return ""

    async def chat_stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Stream the LLM response token by token (SSE).

        Yields content fragments as they arrive from the API.
        Yields nothing (ends immediately) if the API key is missing.
        """
        if not self._api_key:
            return

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{self._base}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                ) as resp:
                    async for raw in resp.aiter_lines():
                        if not raw.startswith("data: "):
                            continue
                        payload = raw[6:]  # strip "data: " prefix
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception:
            logger.exception("LLM streaming API call failed (%s)", self._provider)
