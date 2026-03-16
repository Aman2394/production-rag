"""BM25 sparse retrieval index using rank_bm25."""

from app.exceptions import RetrievalError


async def bm25_search(query: str, top_k: int) -> list[dict]:
    """Retrieve the top-k chunks matching the query using BM25.

    Args:
        query: The raw query string.
        top_k: Number of results to return.

    Returns:
        List of chunk dicts with ``chunk_id``, ``content``, ``score``, and ``metadata``.

    Raises:
        RetrievalError: If the BM25 index is not loaded or the query fails.
    """
    # TODO: implement using rank_bm25.BM25Okapi; persist/load index to disk
    raise RetrievalError("BM25 store not yet implemented.")
