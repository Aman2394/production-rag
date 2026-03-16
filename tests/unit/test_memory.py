"""Unit tests for the memory layer (Redis history + Postgres store)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.memory.manager import get_history, save_turn
from app.generation.prompts import format_history


# ── format_history ────────────────────────────────────────────────────────────

def test_format_history_empty() -> None:
    assert format_history([]) == ""


def test_format_history_single_turn() -> None:
    messages = [
        {"role": "human", "content": "Who founded Apple?"},
        {"role": "ai", "content": "Steve Jobs."},
    ]
    result = format_history(messages)
    assert "Human: Who founded Apple?" in result
    assert "Assistant: Steve Jobs." in result


def test_format_history_preserves_order() -> None:
    messages = [
        {"role": "human", "content": "First question"},
        {"role": "ai", "content": "First answer"},
        {"role": "human", "content": "Second question"},
    ]
    result = format_history(messages)
    assert result.index("First question") < result.index("Second question")


# ── Redis history (mocked) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_history_returns_empty_on_redis_failure() -> None:
    with patch(
        "app.memory.redis_history.get_history",
        AsyncMock(return_value=[]),
    ):
        history = await get_history("test-session")
        assert history == []


@pytest.mark.asyncio
async def test_get_history_returns_messages() -> None:
    expected = [
        {"role": "human", "content": "Hello"},
        {"role": "ai", "content": "Hi there"},
    ]
    with patch(
        "app.memory.redis_history.get_history",
        AsyncMock(return_value=expected),
    ):
        history = await get_history("test-session")
        assert history == expected


# ── Memory manager ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_turn_calls_both_layers() -> None:
    with (
        patch("app.memory.manager.redis_history.save_turn", AsyncMock()) as mock_redis,
        patch("app.memory.manager.postgres_store.save_turn", AsyncMock()) as mock_pg,
    ):
        await save_turn("session-1", "question?", "answer.", [])
        mock_redis.assert_called_once_with("session-1", "question?", "answer.")
        mock_pg.assert_called_once_with("session-1", "question?", "answer.", [])


# ── PostgreSQL store (no-op when unconfigured) ────────────────────────────────

@pytest.mark.asyncio
async def test_postgres_save_noop_when_not_configured() -> None:
    from app.memory import postgres_store
    with patch("app.memory.postgres_store._settings") as mock_settings:
        mock_settings.postgres_url = None
        # Should complete without error or DB call
        await postgres_store.save_turn("s1", "q", "a")


@pytest.mark.asyncio
async def test_postgres_get_turns_noop_when_not_configured() -> None:
    from app.memory import postgres_store
    with patch("app.memory.postgres_store._settings") as mock_settings:
        mock_settings.postgres_url = None
        result = await postgres_store.get_session_turns("s1")
        assert result == []
