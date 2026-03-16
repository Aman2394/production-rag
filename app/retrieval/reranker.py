"""Cross-encoder reranking via Cohere Rerank API."""

import structlog

from app.config import get_settings
from app.exceptions import RerankerError

logger = structlog.get_logger(__name__)
_settings = get_settings()


async def rerank(query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
    """Rerank candidate chunks using the Cohere Rerank API.

    Args:
        query: The user's original query string.
        candidates: List of chunk dicts (must contain ``content`` and ``chunk_id``).
        top_n: Number of chunks to return after reranking; defaults to ``settings.rerank_top_n``.

    Returns:
        Reranked list of chunk dicts, ordered by relevance descending.

    Raises:
        RerankerError: If the Cohere API call fails.
    """
    # TODO: implement using cohere.AsyncClientV2.rerank
    raise RerankerError("Reranker not yet implemented.")
