"""Cross-encoder reranking using BAAI/bge-reranker-v2-m3 (local, no API key).

The CrossEncoder model is loaded once as a singleton and reused across calls.
Scoring runs in a thread-pool executor so it doesn't block the async event loop.
"""

import asyncio
from functools import lru_cache

import structlog
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.exceptions import RerankerError

logger = structlog.get_logger(__name__)
_settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Load and cache the CrossEncoder model (runs once on first call).

    Returns:
        Loaded CrossEncoder model on the configured device.
    """
    logger.info("reranker.model_loading", model=_settings.reranker_model)
    model = CrossEncoder(
        _settings.reranker_model,
        device=_settings.reranker_device,
    )
    logger.info("reranker.model_ready", model=_settings.reranker_model)
    return model


def _score_sync(query: str, passages: list[str]) -> list[float]:
    """Synchronous scoring — called in a thread-pool executor.

    Args:
        query: The user's query string.
        passages: List of candidate passage texts.

    Returns:
        List of relevance scores, one per passage.
    """
    model = _get_model()
    pairs = [[query, passage] for passage in passages]
    scores: list[float] = model.predict(pairs).tolist()
    return scores


async def rerank(
    query: str,
    candidates: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """Rerank candidate chunks using the BGE cross-encoder model.

    Scores every (query, chunk) pair, sorts by relevance descending,
    and returns the top-n chunks.

    Args:
        query: The user's original query string.
        candidates: List of chunk dicts (must contain ``content`` and ``chunk_id``).
        top_n: Number of chunks to return after reranking;
            defaults to ``settings.rerank_top_n``.

    Returns:
        Reranked list of chunk dicts with an added ``rerank_score`` field,
        ordered by relevance descending.

    Raises:
        RerankerError: If scoring fails.
    """
    if not candidates:
        return []

    n = top_n or _settings.rerank_top_n
    passages = [c["content"] for c in candidates]

    logger.info("reranker.start", candidates=len(candidates), top_n=n)
    try:
        loop = asyncio.get_event_loop()
        scores: list[float] = await loop.run_in_executor(
            None, _score_sync, query, passages
        )
    except Exception as exc:
        raise RerankerError(f"Reranking failed: {exc}") from exc

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = [
        {**chunk, "rerank_score": round(score, 6)}
        for score, chunk in ranked[:n]
    ]

    logger.info("reranker.complete", returned=len(results))
    return results
