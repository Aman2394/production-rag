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

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key")
    anthropic_model: str = "claude-sonnet-4-6"

    # ── Embeddings (local sentence-transformers) ──────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimensions: int = 1024  # BGE-large-en-v1.5 output size
    embedding_device: str = "cpu"     # "cpu" | "cuda" | "mps"

    # Optional — only needed if switching back to OpenAI embeddings
    openai_api_key: SecretStr | None = None

    # ── Reranker ──────────────────────────────────────────────────────────────
    cohere_api_key: SecretStr = Field(..., description="Cohere API key for reranking")
    cohere_rerank_model: str = "rerank-english-v3.0"

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Returns:
        Settings: Validated settings instance loaded from the environment.
    """
    return Settings()
