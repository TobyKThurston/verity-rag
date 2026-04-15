"""Settings loaded from VERITY_* environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERITY_", env_file=".env", extra="ignore")

    # generation / embeddings (local via Ollama)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 384

    # retrieval
    retrieval_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 5

    # agent
    max_agent_steps: int = 4

    # storage: in-memory store is used when no DSN is set
    pg_dsn: str | None = Field(default=None)

    # observability
    otel_console_export: bool = True
    service_name: str = "verity"


@lru_cache
def get_settings() -> Settings:
    return Settings()
