"""Memory manager — coordinates short-term (Redis) and long-term (PostgreSQL) storage."""

import asyncio

import structlog

from app.memory import redis_history, postgres_store

logger = structlog.get_logger(__name__)


async def get_history(session_id: str) -> list[dict[str, str]]:
    """Load recent conversation history for a session.

    Reads from Redis (short-term). Falls back to empty list if Redis
    is unavailable.

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        List of message dicts ordered oldest → newest, each with
        ``role`` and ``content`` keys.
    """
    return await redis_history.get_history(session_id)


async def save_turn(
    session_id: str,
    question: str,
    answer: str,
    citations: list[dict] | None = None,
) -> None:
    """Persist a conversation turn to both memory layers concurrently.

    Writes to Redis (short-term, always) and PostgreSQL (long-term,
    only if ``POSTGRES_URL`` is configured) in parallel.

    Args:
        session_id: Unique identifier for the conversation session.
        question: The user's question for this turn.
        answer: The assistant's grounded answer.
        citations: Optional list of citation dicts serialised for long-term storage.
    """
    await asyncio.gather(
        redis_history.save_turn(session_id, question, answer),
        postgres_store.save_turn(session_id, question, answer, citations),
    )
    logger.info("memory.turn_saved", session_id=session_id)
