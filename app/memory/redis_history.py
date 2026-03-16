"""Short-term conversation memory backed by Redis.

Stores the last N messages per session as a Redis List with a TTL.
Each message is a JSON-serialised dict: {"role": "human"|"ai", "content": str}.
"""

import json
from typing import Any

import structlog
import redis.asyncio as aioredis

from app.config import get_settings

logger = structlog.get_logger(__name__)
_settings = get_settings()

# Redis key pattern for session history
_KEY_PREFIX = "rag:history:"


def _session_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _get_client() -> aioredis.Redis:
    return aioredis.from_url(
        _settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )


async def get_history(session_id: str) -> list[dict[str, str]]:
    """Load the recent conversation history for a session from Redis.

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        List of message dicts ordered oldest → newest, each with
        ``role`` ("human" or "ai") and ``content`` keys.
        Returns an empty list if the session does not exist or Redis
        is unavailable.
    """
    try:
        client = _get_client()
        raw: list[str] = await client.lrange(
            _session_key(session_id), 0, _settings.redis_max_history - 1
        )
        await client.aclose()
        # Redis List is newest-first (LPUSH), so reverse for chronological order
        messages: list[dict[str, str]] = [json.loads(m) for m in reversed(raw)]
        logger.debug("redis_history.loaded", session_id=session_id, messages=len(messages))
        return messages
    except Exception as exc:
        logger.warning("redis_history.load_failed", session_id=session_id, error=str(exc))
        return []


async def save_turn(
    session_id: str,
    question: str,
    answer: str,
) -> None:
    """Append a human/AI turn to the Redis history for a session.

    The list is capped at ``settings.redis_max_history`` entries and the
    key is given a TTL of ``settings.redis_history_ttl`` seconds.

    Args:
        session_id: Unique identifier for the conversation session.
        question: The user's question for this turn.
        answer: The assistant's answer for this turn.
    """
    try:
        client = _get_client()
        key = _session_key(session_id)
        pipe = client.pipeline()

        # LPUSH keeps newest first; we reverse on read for chronological order
        pipe.lpush(key, json.dumps({"role": "ai", "content": answer}))
        pipe.lpush(key, json.dumps({"role": "human", "content": question}))
        pipe.ltrim(key, 0, _settings.redis_max_history - 1)
        pipe.expire(key, _settings.redis_history_ttl)

        await pipe.execute()
        await client.aclose()
        logger.debug("redis_history.saved", session_id=session_id)
    except Exception as exc:
        # Non-fatal — log and continue without saving
        logger.warning("redis_history.save_failed", session_id=session_id, error=str(exc))


async def clear_history(session_id: str) -> None:
    """Delete the conversation history for a session.

    Args:
        session_id: Unique identifier for the conversation session.
    """
    try:
        client = _get_client()
        await client.delete(_session_key(session_id))
        await client.aclose()
        logger.info("redis_history.cleared", session_id=session_id)
    except Exception as exc:
        logger.warning("redis_history.clear_failed", session_id=session_id, error=str(exc))
