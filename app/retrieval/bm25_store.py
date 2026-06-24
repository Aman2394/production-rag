"""BM25 sparse retrieval — one isolated index per namespace.

Each namespace gets its own BM25Okapi instance and corpus, persisted to
``data/bm25_indexes/<namespace>.pkl``. Indexes are loaded lazily on first
access, so startup cost is zero for namespaces that haven't been queried yet.
"""

import asyncio
import pickle
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from app.exceptions import RetrievalError

logger = structlog.get_logger(__name__)

_INDEX_DIR = Path("data/bm25_indexes")

# namespace → {"corpus": list[dict], "bm25": BM25Okapi | None}
_indexes: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()  # serialises writes; reads are lock-free (safe in asyncio)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _index_path(namespace: str) -> Path:
    return _INDEX_DIR / f"{namespace}.pkl"


def _rebuild(namespace: str) -> None:
    """Rebuild the BM25Okapi index for a namespace from its in-memory corpus."""
    corpus = _indexes[namespace]["corpus"]
    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    _indexes[namespace]["bm25"] = BM25Okapi(tokenized) if tokenized else None


def _save(namespace: str) -> None:
    """Persist a namespace's corpus to disk (runs in executor thread)."""
    path = _index_path(namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"corpus": _indexes[namespace]["corpus"]}, f)
    logger.info("bm25_store.saved", namespace=namespace, path=str(path))


def _load(namespace: str) -> None:
    """Load a namespace's corpus from disk and rebuild its BM25 index."""
    path = _index_path(namespace)
    if not path.exists():
        return
    with path.open("rb") as f:
        data: dict[str, Any] = pickle.load(f)
    _indexes[namespace]["corpus"] = data.get("corpus", [])
    _rebuild(namespace)
    logger.info(
        "bm25_store.loaded",
        namespace=namespace,
        docs=len(_indexes[namespace]["corpus"]),
    )


def _ensure_namespace(namespace: str) -> None:
    """Guarantee the namespace entry exists in _indexes, loading from disk if needed.

    Safe to call without the lock because:
    - No await points → runs atomically in asyncio.
    - Worst case: two concurrent callers both init the same namespace; the second
      overwrites with identical data (idempotent).
    """
    if namespace not in _indexes:
        _indexes[namespace] = {"corpus": [], "bm25": None}
        try:
            _load(namespace)
        except Exception as exc:
            logger.warning("bm25_store.load_failed", namespace=namespace, error=str(exc))


# ── Public API ────────────────────────────────────────────────────────────────

async def add_chunks(chunks: list[dict], namespace: str) -> None:
    """Add chunks to a namespace's BM25 index and persist to disk.

    Args:
        chunks: List of chunk dicts, each with ``chunk_id``, ``content``,
            and ``metadata``.
        namespace: Target namespace.
    """
    async with _lock:
        _ensure_namespace(namespace)
        _indexes[namespace]["corpus"].extend(chunks)
        _rebuild(namespace)
        # Persist in a thread so we don't block the event loop
        await asyncio.get_event_loop().run_in_executor(None, _save, namespace)
    logger.info(
        "bm25_store.added",
        namespace=namespace,
        added=len(chunks),
        total=len(_indexes[namespace]["corpus"]),
    )


async def bm25_search(query: str, top_k: int, namespace: str) -> list[dict]:
    """Retrieve top-k chunks matching the query in a namespace using BM25.

    Args:
        query: Raw query string.
        top_k: Number of results to return.
        namespace: Namespace to search within.

    Returns:
        Chunks ordered by BM25 score descending.

    Raises:
        RetrievalError: If the namespace has no indexed documents.
    """
    _ensure_namespace(namespace)
    data = _indexes[namespace]
    bm25 = data["bm25"]
    corpus = data["corpus"]

    if bm25 is None or not corpus:
        raise RetrievalError(
            f"BM25 index is empty for namespace '{namespace}'. "
            "Ingest documents into this namespace before querying."
        )

    tokens = _tokenize(query)
    scores: list[float] = bm25.get_scores(tokens).tolist()

    ranked = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)

    return [
        {
            "chunk_id": doc["chunk_id"],
            "content": doc["content"],
            "score": score,
            "metadata": doc.get("metadata", {}),
        }
        for score, doc in ranked[:top_k]
    ]


async def delete_namespace(namespace: str) -> int:
    """Delete all BM25 data for a namespace (in-memory and on disk).

    Args:
        namespace: Namespace to delete.

    Returns:
        Number of chunks that were deleted.
    """
    async with _lock:
        data = _indexes.pop(namespace, {"corpus": []})
        deleted = len(data["corpus"])
        path = _index_path(namespace)
        if path.exists():
            path.unlink()
    logger.info("bm25_store.deleted", namespace=namespace, chunks=deleted)
    return deleted
