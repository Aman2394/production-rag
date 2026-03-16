"""Unit tests for the generation chain."""

from unittest.mock import AsyncMock, patch

import pytest

from app.exceptions import CitationError, GenerationError
from app.generation.chain import _build_citations, _format_context, generate

_UUID1 = "550e8400-e29b-41d4-a716-446655440000"
_UUID2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

_CHUNKS = [
    {
        "chunk_id": _UUID1,
        "content": "The sky is blue.",
        "metadata": {"source": "doc.pdf", "page": 1},
    },
    {
        "chunk_id": _UUID2,
        "content": "Water boils at 100°C.",
        "metadata": {"source": "doc.pdf", "page": 2},
    },
]


def test_format_context_includes_chunk_ids() -> None:
    context = _format_context(_CHUNKS)
    assert _UUID1 in context
    assert _UUID2 in context
    assert "The sky is blue." in context


def test_format_context_separates_chunks() -> None:
    context = _format_context(_CHUNKS)
    assert "---" in context


def test_build_citations_maps_ids_to_metadata() -> None:
    citations = _build_citations([_UUID1], _CHUNKS)
    assert len(citations) == 1
    assert citations[0].chunk_id == _UUID1
    assert citations[0].source == "doc.pdf"
    assert citations[0].page == 1


def test_build_citations_skips_unknown_ids() -> None:
    citations = _build_citations(["unknown-id"], _CHUNKS)
    assert citations == []


@pytest.mark.asyncio
async def test_generate_returns_answer_and_citations() -> None:
    valid_answer = f"The sky is blue [{_UUID1}]."

    with patch(
        "app.generation.chain._invoke_chain",
        AsyncMock(return_value=valid_answer),
    ):
        answer, citations = await generate("What color is the sky?", _CHUNKS)

    assert answer == valid_answer
    assert len(citations) == 1
    assert citations[0].chunk_id == _UUID1
    assert citations[0].source == "doc.pdf"


@pytest.mark.asyncio
async def test_generate_raises_when_no_chunks() -> None:
    with pytest.raises(GenerationError, match="No chunks provided"):
        await generate("Any question?", [])


@pytest.mark.asyncio
async def test_generate_retries_on_citation_error() -> None:
    valid_answer = f"The sky is blue [{_UUID1}]."
    # A well-formed UUID that is NOT in _CHUNKS — triggers CitationError
    _UNKNOWN_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    invalid_answer = f"The sky is blue [{_UNKNOWN_UUID}]."

    call_count = 0

    async def side_effect(context: str, question: str, history: str | None = None) -> str:
        nonlocal call_count
        call_count += 1
        return invalid_answer if call_count == 1 else valid_answer

    with patch("app.generation.chain._invoke_chain", side_effect=side_effect):
        answer, citations = await generate("What color is the sky?", _CHUNKS)

    assert call_count == 2
    assert valid_answer == answer


@pytest.mark.asyncio
async def test_generate_raises_generation_error_after_max_retries() -> None:
    # Always return an answer that cites an ID not in the context
    invalid_answer = f"Answer [{_UUID1}] but this id is not in chunks."
    single_chunk = [
        {
            "chunk_id": _UUID2,  # only UUID2 is valid
            "content": "Some content",
            "metadata": {"source": "x.pdf"},
        }
    ]

    with patch(
        "app.generation.chain._invoke_chain",
        AsyncMock(return_value=invalid_answer),
    ):
        with pytest.raises(GenerationError, match="invalid citations"):
            await generate("question?", single_chunk)


@pytest.mark.asyncio
async def test_generate_raises_when_no_api_key() -> None:
    with patch("app.generation.chain._settings") as mock_settings:
        mock_settings.anthropic_api_key = None
        with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
            await generate("question?", _CHUNKS)
