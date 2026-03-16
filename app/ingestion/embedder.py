"""Embedding model wrapper — local BGE model via sentence-transformers.

The SentenceTransformer model is loaded once at module level and reused
across all calls. Encoding runs in a thread-pool executor so it doesn't
block the async event loop.
"""

import asyncio
from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)

_settings = get_settings()

# Batch size for encoding — tune up if GPU is available
_BATCH_SIZE = 64


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model (runs once on first call).

    Returns:
        Loaded SentenceTransformer model on the configured device.
    """
    logger.info("embedder.model_loading", model=_settings.embedding_model)
    model = SentenceTransformer(
        _settings.embedding_model,
        device=_settings.embedding_device,
    )
    logger.info("embedder.model_ready", model=_settings.embedding_model)
    return model


def _encode_sync(texts: list[str]) -> list[list[float]]:
    """Synchronous encode — called in a thread-pool executor.

    Args:
        texts: List of texts to encode.

    Returns:
        List of embedding vectors as plain Python floats.
    """
    model = _get_model()
    # normalize_embeddings=True gives unit vectors — better for cosine similarity
    vectors = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the local BGE sentence-transformers model.

    Offloads the CPU-bound encoding to a thread-pool executor so the async
    event loop stays unblocked.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (unit-normalized), one per input text,
        in the same order.

    Raises:
        EmbeddingError: If encoding fails.
    """
    if not texts:
        return []

    logger.info("embedder.start", total=len(texts))
    try:
        loop = asyncio.get_event_loop()
        embeddings: list[list[float]] = await loop.run_in_executor(
            None, _encode_sync, texts
        )
        logger.info("embedder.complete", total=len(embeddings))
        return embeddings
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc
