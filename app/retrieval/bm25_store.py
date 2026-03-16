"""BM25 sparse retrieval index using rank_bm25.

The index is held in memory as a module-level singleton and persisted to disk
as a pickle file so it survives restarts without re-ingesting documents.
"""

import asyncio
import pickle
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from app.exceptions import RetrievalError

logger = structlog.get_logger(__name__)

_INDEX_PATH = Path("data/bm25_index.pkl")

# Module-level state: corpus metadata and the BM25 index
_corpus: list[dict] = []        # [{chunk_id, content, metadata}, ...]
_bm25: BM25Okapi | None = None
_lock = asyncio.Lock()


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _rebuild_index() -> None:
    """Rebuild the BM25Okapi index from the current _corpus."""
    global _bm25
    tokenized = [_tokenize(doc["content"]) for doc in _corpus]
    _bm25 = BM25Okapi(tokenized) if tokenized else None


def _save_to_disk() -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _INDEX_PATH.open("wb") as f:
        pickle.dump({"corpus": _corpus}, f)
    logger.info("bm25_store.saved", path=str(_INDEX_PATH), docs=len(_corpus))


def _load_from_disk() -> None:
    global _corpus
    if not _INDEX_PATH.exists():
        return
    with _INDEX_PATH.open("rb") as f:
        data: dict[str, Any] = pickle.load(f)
    _corpus = data.get("corpus", [])
    _rebuild_index()
    logger.info("bm25_store.loaded", path=str(_INDEX_PATH), docs=len(_corpus))


# Load persisted index at import time
try:
    _load_from_disk()
except Exception as _exc:
    logger.warning("bm25_store.load_failed", error=str(_exc))


async def add_chunks(chunks: list[dict]) -> None:
    """Add chunks to the BM25 index and persist to disk.

    Args:
        chunks: List of chunk dicts, each containing ``chunk_id``,
            ``content``, and ``metadata``.
    """
    async with _lock:
        _corpus.extend(chunks)
        _rebuild_index()
        await asyncio.get_event_loop().run_in_executor(None, _save_to_disk)
    logger.info("bm25_store.added", added=len(chunks), total=len(_corpus))


async def bm25_search(query: str, top_k: int) -> list[dict]:
    """Retrieve the top-k chunks matching the query using BM25.

    Args:
        query: The raw query string.
        top_k: Number of results to return.

    Returns:
        List of chunk dicts with ``chunk_id``, ``content``, ``score``,
        and ``metadata``, ordered by BM25 score descending.

    Raises:
        RetrievalError: If the BM25 index is empty.
    """
    if _bm25 is None or not _corpus:
        raise RetrievalError(
            "BM25 index is empty. Ingest documents before querying."
        )

    tokens = _tokenize(query)
    scores: list[float] = _bm25.get_scores(tokens).tolist()

    ranked = sorted(
        zip(scores, _corpus),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        {
            "chunk_id": doc["chunk_id"],
            "content": doc["content"],
            "score": score,
            "metadata": doc.get("metadata", {}),
        }
        for score, doc in ranked[:top_k]
    ]
