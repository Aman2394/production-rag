"""Full retrieval orchestration: embed query → dense + sparse → RRF → rerank."""

import structlog

from app.config import get_settings
from app.exceptions import RetrievalError
from app.ingestion.embedder import embed_texts
from app.retrieval.bm25_store import bm25_search
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import similarity_search

logger = structlog.get_logger(__name__)
_settings = get_settings()


async def retrieve(query: str, top_n: int | None = None) -> list[dict]:
    """Run the full hybrid retrieval pipeline for a query.

    Embeds the query, performs dense and sparse retrieval in parallel,
    fuses results with RRF, then reranks with a cross-encoder.

    Args:
        query: The user's natural language question.
        top_n: Final number of chunks to return; defaults to ``settings.rerank_top_n``.

    Returns:
        Reranked list of the most relevant chunk dicts.

    Raises:
        RetrievalError: If any stage of the pipeline fails.
    """
    import asyncio

    logger.info("retrieval.start", query=query)
    try:
        query_vector, sparse = await asyncio.gather(
            embed_texts([query]),
            bm25_search(query, top_k=_settings.retrieval_top_k),
        )
        dense = await similarity_search(query_vector[0], top_k=_settings.retrieval_top_k)
        fused = reciprocal_rank_fusion(dense, sparse, top_k=_settings.retrieval_top_k)
        results = await rerank(query, fused, top_n=top_n or _settings.rerank_top_n)
        logger.info("retrieval.complete", chunks_returned=len(results))
        return results
    except Exception as exc:
        raise RetrievalError(f"Retrieval pipeline failed: {exc}") from exc
