"""Unit tests for RRF hybrid fusion."""

import pytest

from app.retrieval.hybrid import reciprocal_rank_fusion


def _make_chunks(ids: list[str]) -> list[dict]:
    return [{"chunk_id": cid, "content": f"content-{cid}", "score": 1.0} for cid in ids]


def test_rrf_returns_top_k() -> None:
    dense = _make_chunks(["a", "b", "c"])
    sparse = _make_chunks(["b", "c", "d"])
    result = reciprocal_rank_fusion(dense, sparse, top_k=2)
    assert len(result) == 2


def test_rrf_boosts_overlapping_chunks() -> None:
    dense = _make_chunks(["a", "b"])
    sparse = _make_chunks(["b", "c"])
    result = reciprocal_rank_fusion(dense, sparse, top_k=3)
    # "b" appears in both lists — it should be ranked first
    assert result[0]["chunk_id"] == "b"


def test_rrf_deduplicates_chunks() -> None:
    dense = _make_chunks(["a", "b"])
    sparse = _make_chunks(["a", "b"])
    result = reciprocal_rank_fusion(dense, sparse, top_k=10)
    ids = [r["chunk_id"] for r in result]
    assert len(ids) == len(set(ids))
