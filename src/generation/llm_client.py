from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

# Retry on throttling / transient failures — Qwen-VL is API-bound and a 429
# would otherwise silently drop a figure/table description (quality loss).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 3

# Magic bytes → MIME type mapping for multimodal image payloads
_MIME_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
]


def _guess_mime(data: bytes) -> str:
    """Return the MIME type of *data* based on magic bytes."""
    for magic, mime in _MIME_MAGIC:
        if data.startswith(magic):
            return mime
    return "image/png"  # safe fallback


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
            self._vl_model = s.qwen_vl_model
        else:
            self._api_key = api_key or s.deepseek_api_key
            self._base = s.deepseek_api_base
            self._model = s.deepseek_model
            self._vl_model = ""

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

    async def chat_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Multimodal chat: send text + an image to a vision-capable model.

        For ``provider="qwen"`` this uses ``qwen_vl_model`` — NOT the text
        ``qwen_model``.  The image is embedded as a ``data:`` URI so the VL
        model can actually see it.

        Returns ``""`` on failure (missing key, API error, empty response).
        """
        if not self._api_key:
            return ""

        model = self._vl_model if self._provider == "qwen" else self._model
        if not model:
            return ""

        import base64

        mime = _guess_mime(image_bytes)
        data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        )

        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{self._base}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Qwen-VL API %s, retry %d/%d in %.1fs",
                        status, attempt + 1, _MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        logger.error("LLM multimodal API call failed after %d retries (%s)", _MAX_RETRIES, self._provider)
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
