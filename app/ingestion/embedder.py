"""Embedding model wrapper (OpenAI text-embedding-3-small)."""

from app.config import get_settings
from app.exceptions import EmbeddingError

_settings = get_settings()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured embedding model.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors, one per input text.

    Raises:
        EmbeddingError: If the embedding API call fails.
    """
    # TODO: implement using openai.AsyncOpenAI with asyncio.gather for batching
    raise EmbeddingError("Embedder not yet implemented.")
