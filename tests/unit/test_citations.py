"""Unit tests for citation extraction and validation."""

import pytest

from app.exceptions import CitationError
from app.generation.citations import extract_cited_ids, validate_citations

_UUID1 = "550e8400-e29b-41d4-a716-446655440000"
_UUID2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


def test_extract_single_citation() -> None:
    response = f"The sky is blue [{_UUID1}]."
    assert extract_cited_ids(response) == [_UUID1]


def test_extract_multiple_citations() -> None:
    response = f"Fact one [{_UUID1}]. Fact two [{_UUID2}]."
    assert extract_cited_ids(response) == [_UUID1, _UUID2]


def test_extract_deduplicates() -> None:
    response = f"Repeated [{_UUID1}] and again [{_UUID1}]."
    assert extract_cited_ids(response) == [_UUID1]


def test_extract_no_citations() -> None:
    assert extract_cited_ids("No citations here.") == []


def test_validate_passes_with_valid_citations() -> None:
    validate_citations([_UUID1], valid_chunk_ids={_UUID1})  # Should not raise


def test_validate_raises_on_invalid_citation() -> None:
    with pytest.raises(CitationError):
        validate_citations([_UUID1], valid_chunk_ids={_UUID2})
