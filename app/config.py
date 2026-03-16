"""Application configuration loaded from environment variables via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings sourced from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ───────────────────────────────────────────────────────────
    env: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    # ── LLM provider: "ollama" (default, local) | "anthropic" ────────────────
    llm_provider: str = "ollama"

    # Ollama (local, no API key required)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Anthropic (optional — set llm_provider=anthropic to use)
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    # ── Embeddings (local sentence-transformers) ──────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimensions: int = 1024  # BGE-large-en-v1.5 output size
    embedding_device: str = "cpu"     # "cpu" | "cuda" | "mps"

    # Optional — only needed if switching back to OpenAI embeddings
    openai_api_key: SecretStr | None = None

    # ── Reranker (local BGE cross-encoder) ───────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"      # "cpu" | "cuda" | "mps"

    # Optional — only needed if switching to Cohere Rerank API
    cohere_api_key: SecretStr | None = None

    # ── Vector Store ──────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "documents"

    # ── Observability ─────────────────────────────────────────────────────────
    langfuse_secret_key: SecretStr | None = None
    langfuse_public_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    langsmith_api_key: SecretStr | None = None
    langchain_tracing_v2: bool = True
    langchain_project: str = "production-rag"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=20, gt=0, description="Candidates sent to reranker")
    rerank_top_n: int = Field(default=5, gt=0, description="Final docs returned to LLM")

    # ── Memory — short-term (Redis, required for conversation history) ─────────
    redis_url: str = "redis://localhost:6379"
    redis_history_ttl: int = Field(default=86400, description="Session TTL in seconds (24h)")
    redis_max_history: int = Field(default=20, description="Max messages kept per session (10 turns)")

    # ── Memory — long-term (PostgreSQL, optional) ─────────────────────────────
    postgres_url: str | None = None  # e.g. postgresql://user:pass@localhost/ragdb


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Returns:
        Settings: Validated settings instance loaded from the environment.
    """
    return Settings()
