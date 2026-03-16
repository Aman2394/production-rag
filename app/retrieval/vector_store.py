"""Qdrant client wrapper for dense vector retrieval."""

from app.config import get_settings
from app.exceptions import RetrievalError

_settings = get_settings()


async def similarity_search(
    query_vector: list[float],
    top_k: int | None = None,
) -> list[dict]:
    """Retrieve the top-k most similar chunks from Qdrant.

    Args:
        query_vector: Dense embedding of the query.
        top_k: Number of results to return; defaults to ``settings.retrieval_top_k``.

    Returns:
        List of chunk dicts with ``chunk_id``, ``content``, ``score``, and ``metadata``.

    Raises:
        RetrievalError: If the Qdrant query fails.
    """
    # TODO: implement using qdrant_client.AsyncQdrantClient
    raise RetrievalError("Vector store not yet implemented.")
