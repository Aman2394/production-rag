"""Unit tests for the ingestion pipeline internals."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.ingestion.pipeline import _build_chunks, run_ingestion


def test_build_chunks_assigns_required_metadata() -> None:
    docs = [Document(page_content="Hello world. " * 20, metadata={"page": 1})]
    chunks = _build_chunks(docs, source="test.pdf")
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "content" in chunk
        assert chunk["metadata"]["source"] == "test.pdf"
        assert chunk["metadata"]["page"] == 1
        assert "ingested_at" in chunk["metadata"]


def test_build_chunks_skips_empty_splits() -> None:
    docs = [Document(page_content="   \n\n   ", metadata={})]
    chunks = _build_chunks(docs, source="empty.md")
    assert chunks == []


def test_build_chunks_unique_chunk_ids() -> None:
    docs = [Document(page_content="Word " * 200, metadata={})]
    chunks = _build_chunks(docs, source="test.md")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_run_ingestion_returns_chunk_count() -> None:
    mock_doc = Document(page_content="Sample content. " * 30, metadata={"page": 1})

    with (
        patch("app.ingestion.pipeline.load_document", AsyncMock(return_value=[mock_doc])),
        patch("app.ingestion.pipeline.embed_texts", AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))),
        patch("app.ingestion.pipeline.upsert_chunks", AsyncMock()),
        patch("app.ingestion.pipeline.add_chunks", AsyncMock()),
    ):
        count = await run_ingestion("test.pdf")
        assert count > 0


@pytest.mark.asyncio
async def test_run_ingestion_raises_on_load_failure() -> None:
    from app.exceptions import IngestionError

    with patch("app.ingestion.pipeline.load_document", AsyncMock(side_effect=IngestionError("not found"))):
        with pytest.raises(IngestionError):
            await run_ingestion("missing.pdf")
