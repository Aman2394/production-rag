"""Reciprocal Rank Fusion (RRF) fusion of dense and sparse retrieval results."""

from app.exceptions import RetrievalError

_RRF_K = 60  # Standard RRF constant


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
) -> list[dict]:
    """Fuse dense and sparse ranked lists using Reciprocal Rank Fusion.

    Args:
        dense_results: Ranked list of chunks from vector search.
        sparse_results: Ranked list of chunks from BM25 search.
        top_k: Number of fused results to return.

    Returns:
        Fused and re-ranked list of chunk dicts, ordered by RRF score descending.

    Raises:
        RetrievalError: If fusion fails due to malformed inputs.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked[:top_k]]
