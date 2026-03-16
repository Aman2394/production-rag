"""Unit tests for the BGE cross-encoder reranker."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.exceptions import RerankerError
from app.retrieval.reranker import rerank


def _make_candidates(ids: list[str]) -> list[dict]:
    return [
        {"chunk_id": cid, "content": f"content about {cid}", "score": 1.0}
        for cid in ids
    ]


@pytest.mark.asyncio
async def test_rerank_returns_top_n() -> None:
    candidates = _make_candidates(["a", "b", "c", "d", "e"])
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.5])

    with patch("app.retrieval.reranker._score_sync", return_value=scores.tolist()):
        results = await rerank("test query", candidates, top_n=3)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_rerank_orders_by_score_descending() -> None:
    candidates = _make_candidates(["a", "b", "c"])
    scores = np.array([0.1, 0.9, 0.5])

    with patch("app.retrieval.reranker._score_sync", return_value=scores.tolist()):
        results = await rerank("test query", candidates, top_n=3)

    assert results[0]["chunk_id"] == "b"
    assert results[1]["chunk_id"] == "c"
    assert results[2]["chunk_id"] == "a"


@pytest.mark.asyncio
async def test_rerank_adds_rerank_score_field() -> None:
    candidates = _make_candidates(["a", "b"])
    scores = np.array([0.4, 0.7])

    with patch("app.retrieval.reranker._score_sync", return_value=scores.tolist()):
        results = await rerank("test query", candidates, top_n=2)

    for r in results:
        assert "rerank_score" in r


@pytest.mark.asyncio
async def test_rerank_empty_candidates_returns_empty() -> None:
    results = await rerank("test query", [], top_n=5)
    assert results == []


@pytest.mark.asyncio
async def test_rerank_raises_on_model_failure() -> None:
    candidates = _make_candidates(["a", "b"])

    with patch("app.retrieval.reranker._score_sync", side_effect=RuntimeError("model error")):
        with pytest.raises(RerankerError, match="Reranking failed"):
            await rerank("test query", candidates, top_n=2)
