"""Qdrant client wrapper for dense vector upsert and retrieval."""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import get_settings
from app.exceptions import RetrievalError

logger = structlog.get_logger(__name__)
_settings = get_settings()


def _get_client() -> AsyncQdrantClient:
    api_key = (
        _settings.qdrant_api_key.get_secret_value() if _settings.qdrant_api_key else None
    )
    return AsyncQdrantClient(url=_settings.qdrant_url, api_key=api_key)


async def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist.

    Raises:
        RetrievalError: If the collection cannot be created.
    """
    client = _get_client()
    try:
        exists = await client.collection_exists(_settings.qdrant_collection_name)
        if not exists:
            await client.create_collection(
                collection_name=_settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=_settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("vector_store.collection_created", name=_settings.qdrant_collection_name)
    except Exception as exc:
        raise RetrievalError(f"Failed to ensure Qdrant collection: {exc}") from exc
    finally:
        await client.close()


async def upsert_chunks(chunks: list[dict]) -> None:
    """Upsert a list of chunks into the Qdrant collection.

    Args:
        chunks: List of chunk dicts, each containing:
            - ``chunk_id`` (str): UUID used as the point ID.
            - ``embedding`` (list[float]): Dense vector.
            - ``content`` (str): Raw text stored in payload.
            - ``metadata`` (dict): Additional payload fields.

    Raises:
        RetrievalError: If the upsert fails.
    """
    if not chunks:
        return

    client = _get_client()
    try:
        await ensure_collection()
        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=chunk["embedding"],
                payload={
                    "content": chunk["content"],
                    **chunk["metadata"],
                },
            )
            for chunk in chunks
        ]
        await client.upsert(
            collection_name=_settings.qdrant_collection_name,
            points=points,
        )
        logger.info("vector_store.upserted", count=len(points))
    except Exception as exc:
        raise RetrievalError(f"Qdrant upsert failed: {exc}") from exc
    finally:
        await client.close()


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
    k = top_k or _settings.retrieval_top_k
    client = _get_client()
    try:
        results = await client.query_points(
            collection_name=_settings.qdrant_collection_name,
            query=query_vector,
            limit=k,
        )
        return [
            {
                "chunk_id": str(hit.id),
                "content": hit.payload.get("content", "") if hit.payload else "",
                "score": hit.score,
                "metadata": {
                    k: v for k, v in (hit.payload or {}).items() if k != "content"
                },
            }
            for hit in results.points
        ]
    except Exception as exc:
        raise RetrievalError(f"Qdrant similarity search failed: {exc}") from exc
    finally:
        await client.close()
