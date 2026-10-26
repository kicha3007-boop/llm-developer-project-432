"""Типизированные настройки приложения — единственное место, где читается окружение."""

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # База данных
    database_url: str = "postgresql://card:card@localhost:5432/card"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    db_pool_timeout_s: float = 10.0

    # Модель (OpenAI-совместимый API)
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str | None = "lm-studio"
    llm_model: str = "qwen2.5-7b-instruct"
    llm_cheap_model: str = "qwen2.5-3b-instruct"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = Field(default=3, ge=1)
    llm_structured_output: Literal["json_schema", "off"] = "json_schema"
    llm_repair_attempts: int = Field(default=2, ge=0, le=5)
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.3

    # Цены за миллион токенов. Деньги — только Decimal.
    llm_price_input_per_1m: Decimal = Decimal("0.20")
    llm_price_output_per_1m: Decimal = Decimal("0.60")
    llm_cheap_price_input_per_1m: Decimal = Decimal("0.05")
    llm_cheap_price_output_per_1m: Decimal = Decimal("0.10")

    # Temporal
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "card-queue"

    # Эмбеддинги
    embedding_model: str = "intfloat/multilingual-e5-base"
    embed_dim: int = 768
    embedding_query_prefix: str = "query: "
    embedding_document_prefix: str = "passage: "
    embedding_batch_size: int = 32

    # Поиск и контекст
    relevance_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    search_top_k: int = 8
    rrf_k: int = 60
    context_max_chars: int = 6000

    # Генерация
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_pipeline_rounds: int = Field(default=3, ge=1, le=5)

    # Документы
    upload_dir: str = "storage/uploads"
    upload_max_bytes: int = 20 * 1024 * 1024
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Защита
    # больше стольких подозрительных фрагментов — задача уходит человеку
    suspicious_chunk_limit: int = 2

    # Логи
    log_json: bool = False
    log_level: str = "INFO"
    log_file: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
