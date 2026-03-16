"""Unit tests for the text chunker."""

import pytest

from app.ingestion.chunker import build_splitter


def test_build_splitter_defaults() -> None:
    splitter = build_splitter()
    assert splitter._chunk_size > 0
    assert splitter._chunk_overlap >= 0


def test_splitter_splits_text() -> None:
    splitter = build_splitter(chunk_size=50, chunk_overlap=5)
    text = "Word " * 100
    chunks = splitter.split_text(text)
    assert len(chunks) > 1


def test_splitter_respects_chunk_size() -> None:
    splitter = build_splitter(chunk_size=100, chunk_overlap=0)
    text = "A" * 500
    chunks = splitter.split_text(text)
    for chunk in chunks:
        assert len(chunk) <= 100
