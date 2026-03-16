"""Long-term conversation storage backed by PostgreSQL (optional).

This layer is only active when ``POSTGRES_URL`` is set in the environment.
All public functions silently no-op if the database is not configured,
so the rest of the application never needs to check for its presence.
"""

import json
from datetime import datetime, timezone

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)
_settings = get_settings()

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    question    TEXT        NOT NULL,
    answer      TEXT        NOT NULL,
    citations   JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations (session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations (created_at);
"""

_INSERT_SQL = """
INSERT INTO conversations (session_id, question, answer, citations, created_at)
VALUES ($1, $2, $3, $4, $5)
"""

_SELECT_SQL = """
SELECT question, answer, created_at
FROM   conversations
WHERE  session_id = $1
ORDER  BY created_at ASC
"""


def _is_configured() -> bool:
    return bool(_settings.postgres_url)


async def _get_pool():  # type: ignore[return]
    """Return an asyncpg connection pool (imported lazily to keep asyncpg optional).

    Returns:
        asyncpg.Pool connected to the configured PostgreSQL URL.
    """
    import asyncpg  # type: ignore[import]
    return await asyncpg.create_pool(_settings.postgres_url)


async def ensure_table() -> None:
    """Create the ``conversations`` table if it does not already exist.

    No-ops if PostgreSQL is not configured.
    """
    if not _is_configured():
        return
    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
        await pool.close()
        logger.info("postgres_store.table_ready")
    except Exception as exc:
        logger.warning("postgres_store.ensure_table_failed", error=str(exc))


async def save_turn(
    session_id: str,
    question: str,
    answer: str,
    citations: list[dict] | None = None,
) -> None:
    """Persist a conversation turn to PostgreSQL.

    No-ops if PostgreSQL is not configured or the insert fails.

    Args:
        session_id: Unique identifier for the conversation session.
        question: The user's question.
        answer: The assistant's grounded answer.
        citations: Optional list of citation dicts for the turn.
    """
    if not _is_configured():
        return
    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                _INSERT_SQL,
                session_id,
                question,
                answer,
                json.dumps(citations or []),
                datetime.now(timezone.utc),
            )
        await pool.close()
        logger.debug("postgres_store.turn_saved", session_id=session_id)
    except Exception as exc:
        # Non-fatal — log and continue
        logger.warning("postgres_store.save_failed", session_id=session_id, error=str(exc))


async def get_session_turns(session_id: str) -> list[dict]:
    """Retrieve all historical turns for a session from PostgreSQL.

    No-ops (returns empty list) if PostgreSQL is not configured.

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        List of turn dicts with ``question``, ``answer``, and ``created_at`` keys,
        ordered chronologically oldest → newest.
    """
    if not _is_configured():
        return []
    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(_SELECT_SQL, session_id)
        await pool.close()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("postgres_store.fetch_failed", session_id=session_id, error=str(exc))
        return []
