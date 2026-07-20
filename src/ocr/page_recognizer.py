"""Scanned PDF page recognizer — Qwen-VL multimodal API for full-page OCR.

For scanned PDF documents that have no extractable text, this module sends
a rendered page image to Qwen-VL and returns structured Markdown output
preserving headings, tables, formulas, and reading order.
"""

from __future__ import annotations

import base64
import io
import logging

import httpx
import numpy as np
from PIL import Image
from src.config import get_settings

logger = logging.getLogger(__name__)

# Page recognition prompt — asks the VL model to produce faithful Markdown
_PAGE_PROMPT = (
    "你是一个文档扫描件识别引擎。请完整识别这张图片中的所有文字内容，"
    "严格按照文档的原始结构和阅读顺序输出 Markdown。\n\n"
    "要求：\n"
    "1. 保留标题层级（用 # / ## / ###）\n"
    "2. 表格用 Markdown 管道符（|）格式输出，保持行列关系\n"
    "3. 数学公式用 $$LaTeX$$ 包裹\n"
    "4. 列表使用 - 或 1. 前缀\n"
    "5. 不要遗漏任何文字内容，不要做摘要或改写\n"
    "6. 如果图片不包含文字，输出空字符串\n"
    "7. 不要输出任何额外说明文字"
)


class PageRecognizer:
    """Full-page OCR recognition using Qwen-VL.

    Usage::

        recognizer = PageRecognizer()
        markdown, confidence = await recognizer.recognize_page(image_bytes)
    """

    def __init__(self):
        self._settings = get_settings()

    async def recognize_page(self, image: np.ndarray | bytes) -> tuple[str, float]:
        """Recognize a full page image and return structured Markdown.

        Args:
            image: Page image as numpy array (H, W, 3) or JPEG/PNG bytes.

        Returns:
            (markdown_text, confidence)
                ``markdown_text`` is empty on failure.
        """
        if not self._settings.qwen_api_key:
            logger.warning("Qwen API key not set — skipping page recognition")
            return "", 0.0

        # Encode the image to JPEG bytes
        img_bytes = self._encode_image(image)

        mime = "image/jpeg"
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{self._settings.qwen_api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._settings.qwen_api_key}"
                    },
                    json={
                        "model": self._settings.qwen_vl_model,
                        "messages": messages,
                        "max_tokens": 4096,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.exception("Qwen-VL page recognition failed")
            return "", 0.0

        if not raw:
            return "", 0.0

        # Estimate confidence based on response length (more content = more likely
        # the model actually recognized something meaningful)
        confidence = min(0.95, 0.5 + len(raw) / 2000)

        return raw.strip(), confidence

    @staticmethod
    def _encode_image(image: np.ndarray | bytes) -> bytes:
        """Convert numpy array or raw bytes to JPEG bytes."""
        if isinstance(image, bytes):
            return image
        # numpy array → PIL → JPEG
        pil_img = Image.fromarray(image.astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
