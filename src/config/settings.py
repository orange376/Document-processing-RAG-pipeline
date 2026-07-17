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
    qwen_vl_model: str = "qwen2.5-vl-3b-instruct"

    # === Paths ===
    upload_dir: str = "./data/uploads"
    vector_db_dir: str = "./data/vector_db"
    model_dir: str = "./data/models"

    # === GPU ===
    device: str = "cuda"  # "cuda" | "cpu"

    # === Confidence Thresholds ===
    confidence_threshold_accept: float = 0.75
    confidence_threshold_reject: float = 0.40

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
