"""Formula recognition engine — Qwen-VL multimodal API.

Recognises mathematical formulas from document-embedded images (e.g. Word
formula screenshots) and returns LaTeX representation.

Falls back gracefully when the API is unavailable.
"""

from __future__ import annotations

import base64
import io
import logging
import struct

import httpx
from src.config import get_settings

logger = logging.getLogger(__name__)

# Magic bytes → MIME type mapping for docx images (PNG / JPEG / GIF / BMP)
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


def _upscale_for_recognition(image_bytes: bytes) -> bytes:
    """Upscale small formula images for better VL model recognition.

    Tiny formula glyphs (e.g. 100×20 px) are hard for VL models to read.
    This scales images to at least 600 px wide while preserving aspect ratio.
    """
    import io

    try:
        from PIL import Image

        im = Image.open(io.BytesIO(image_bytes))
        w, h = im.size
        if w >= 600:
            return image_bytes  # already large enough
        scale = max(2, 600 // max(w, 1) + 1)
        new_w, new_h = w * scale, h * scale
        rgb = im.convert("RGB")
        resized = rgb.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_bytes


class FormulaRecognizer:
    """Recognise math formulas from image bytes using Qwen-VL.

    Usage::

        recognizer = FormulaRecognizer()
        latex, confidence = recognizer.recognize(image_bytes)
    """

    def __init__(self):
        self._settings = get_settings()
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def recognize(self, image_bytes: bytes) -> tuple[str, float]:
        """Recognise a formula from raw image bytes.

        Returns
        -------
        (latex_string, confidence)
            ``latex_string`` is empty on failure.
        """
        if not self._settings.qwen_api_key:
            logger.warning("Qwen API key not set — skipping formula recognition")
            return "", 0.0

        # Upscale tiny formula images before sending to VL model
        image_bytes = _upscale_for_recognition(image_bytes)

        mime = _guess_mime(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请将图片中的数学公式精确转换为LaTeX代码。\n"
                            "要求：\n"
                            "1. 只输出LaTeX代码，用$$包裹\n"
                            "2. 仔细辨认每个符号、上下标、分式、根号、积分、求和等\n"
                            "3. 不要添加任何解释文字\n"
                            "4. 对于分式用 \\frac{分子}{分母}\n"
                            "5. 对于上下标用 ^ 和 _\n"
                            "6. 如果图片不是数学公式，输出 NOT_FORMULA"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self._settings.qwen_api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._settings.qwen_api_key}"
                    },
                    json={
                        "model": self._settings.qwen_vl_model,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.exception("Qwen-VL formula recognition failed")
            return "", 0.0

        # Clean up the response — strip markdown fences if the model added them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("\n", 1)[0] if "\n" in raw else raw[:-3]

        # If the model returned an empty / non-formula response
        if not raw or raw == "``````":
            return "", 0.0

        # Confidence: we trust the VL model for formula images
        confidence = 0.85 if raw.startswith("$$") else 0.7

        return raw.strip(), confidence
