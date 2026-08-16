from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === LLM: DeepSeek (主语言模型) ===
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # === LLM: Qwen (多模态兜底) ===
    qwen_api_key: str = ""
    qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3-7b-plus"
    qwen_vl_model: str = "qwen-vl-plus"  # good balance of accuracy/speed for formula OCR

    # === Paths ===
    upload_dir: str = "./data/uploads"
    vector_db_dir: str = "./data/vector_db"
    model_dir: str = "./data/models"

    # === Vector Store ===
    qdrant_collection: str = "documents_v2"  # bge-base-zh 768 维对应 collection
    embedding_dim: int = 768
    # Qdrant server URL — empty = local file mode; set (e.g. http://qdrant:6333
    # in Docker, or http://localhost:6333 for a standalone server) for
    # multi-process / higher-concurrency access.
    qdrant_url: str = ""

    # === GPU ===
    device: str = "cuda"  # "cuda" | "cpu"

    # === 启动内存 ===
    # 预热 embed + reranker 模型（常驻内存 +1~2GB + 显存 1.6GB）。
    # false = 省内存（首次请求现场加载模型，慢 3-5s）；true = 首次查询快。
    warmup_models: bool = False

    # === 并发处理 ===
    # 同时处理的文档数上限。超出部分进入 FIFO 排队（"排队中"状态）。
    # 限制并发的目的是避免多个文档同时加载 PP-DocLayoutV3 / easyocr /
    # embedding 模型导致内存/显存超限——每套模型约 1~2GB 内存 + GPU。
    max_concurrent_processing: int = 2

    # === Confidence Thresholds ===
    confidence_threshold_accept: float = 0.75
    confidence_threshold_reject: float = 0.40

    # === Context Window ===
    max_context_tokens: int = 3600  # max tokens fed into LLM context (leaves headroom for prompt + answer)

    # === Auth ===
    api_auth_token: str = ""  # Bearer token for API auth; empty = disabled (open mode)

    # === Redis ===
    redis_url: str = "redis://localhost:6379/0"

    @property
    def resolved_upload_dir(self) -> Path:
        return Path(self.upload_dir).resolve()

    @property
    def resolved_vector_db_dir(self) -> Path:
        return Path(self.vector_db_dir).resolve()

    @property
    def resolved_model_dir(self) -> Path:
        return Path(self.model_dir).resolve()


@lru_cache()
def get_settings() -> Settings:
    return Settings()
