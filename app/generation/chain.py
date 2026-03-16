"""Generation pipeline: context formatting → LLM → citation validation + retry.

Supports optional conversation memory via session_id. When history is present:
  1. The follow-up question is contextualised into a standalone retrieval query.
  2. The history is included in the generation prompt.
  3. The completed turn is saved to Redis (short-term) and optionally PostgreSQL.
"""

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

from app.config import get_settings
from app.exceptions import CitationError, GenerationError
from app.generation.citations import extract_cited_ids, validate_citations
from app.generation.prompts import (
    CONTEXTUALIZE_PROMPT,
    RAG_PROMPT,
    RAG_PROMPT_WITH_HISTORY,
    format_history,
)
from app.api.schemas import Citation

logger = structlog.get_logger(__name__)
_settings = get_settings()

_MAX_RETRIES = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into the context block for the prompt.

    Args:
        chunks: List of chunk dicts containing ``chunk_id`` and ``content``.

    Returns:
        Formatted context string with chunk IDs as headers.
    """
    parts = [f"[{c['chunk_id']}]\n{c['content']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def _build_citations(cited_ids: list[str], chunks: list[dict]) -> list[Citation]:
    """Build Citation objects from cited IDs and chunk metadata.

    Args:
        cited_ids: List of chunk IDs extracted from the LLM response.
        chunks: Retrieved chunk dicts containing metadata.

    Returns:
        List of Citation objects for all cited chunk IDs found in chunks.
    """
    chunk_map = {c["chunk_id"]: c for c in chunks}
    citations = []
    for cid in cited_ids:
        chunk = chunk_map.get(cid)
        if chunk:
            meta = chunk.get("metadata", {})
            citations.append(
                Citation(
                    chunk_id=cid,
                    source=meta.get("source", "unknown"),
                    page=meta.get("page"),
                )
            )
    return citations


def _get_llm() -> ChatAnthropic:
    if not _settings.anthropic_api_key:
        raise GenerationError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file."
        )
    return ChatAnthropic(
        model=_settings.anthropic_model,
        api_key=_settings.anthropic_api_key.get_secret_value(),
    )


async def _invoke_chain(
    context: str,
    question: str,
    history: str | None = None,
) -> str:
    """Build and invoke the LCEL chain for a single question.

    Selects the history-aware prompt when history is provided.
    Extracted as a standalone function so it can be patched in tests.

    Args:
        context: Formatted context string from retrieved chunks.
        question: The user's question (may include retry reminder).
        history: Optional formatted conversation history string.

    Returns:
        Raw answer string from the LLM.

    Raises:
        GenerationError: If the LLM call fails.
    """
    llm = _get_llm()
    if history:
        chain = RAG_PROMPT_WITH_HISTORY | llm | StrOutputParser()
        inputs = {"context": context, "question": question, "history": history}
    else:
        chain = RAG_PROMPT | llm | StrOutputParser()
        inputs = {"context": context, "question": question}

    try:
        return await chain.ainvoke(inputs)
    except Exception as exc:
        raise GenerationError(f"LLM call failed: {exc}") from exc


async def contextualize_question(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """Reformulate a follow-up question into a standalone retrieval query.

    Calls the LLM with the conversation history to produce a self-contained
    question suitable for vector and BM25 retrieval.

    Args:
        question: The raw follow-up question from the user.
        history: Conversation history as a list of message dicts.

    Returns:
        Standalone question string. Falls back to the original question
        if the LLM call fails.
    """
    if not history:
        return question
    try:
        llm = _get_llm()
        chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
        standalone: str = await chain.ainvoke(
            {"history": format_history(history), "question": question}
        )
        logger.info(
            "generation.contextualised",
            original=question,
            standalone=standalone,
        )
        return standalone.strip()
    except Exception as exc:
        logger.warning("generation.contextualise_failed", error=str(exc))
        return question


# ── Main entry point ──────────────────────────────────────────────────────────

async def generate(
    question: str,
    chunks: list[dict],
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[Citation]]:
    """Run the RAG generation pipeline with citation validation and retry.

    Formats the retrieved chunks as context, optionally includes conversation
    history, invokes the Claude LLM via the LCEL chain, validates citations,
    and retries once if citation validation fails.

    Args:
        question: The user's natural language question.
        chunks: Reranked list of chunk dicts from the retrieval pipeline.
        history: Optional conversation history (list of role/content dicts).

    Returns:
        Tuple of (answer string, list of Citation objects).

    Raises:
        GenerationError: If the LLM call fails or citations remain invalid
            after all retries.
    """
    if not chunks:
        raise GenerationError("No chunks provided — run retrieval before generation.")

    context = _format_context(chunks)
    valid_ids = {c["chunk_id"] for c in chunks}
    history_str = format_history(history) if history else None

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        logger.info("generation.attempt", attempt=attempt, question=question)
        try:
            answer: str = await _invoke_chain(context, question, history=history_str)
            cited_ids = extract_cited_ids(answer)
            validate_citations(cited_ids, valid_ids)
            citations = _build_citations(cited_ids, chunks)
            logger.info(
                "generation.complete",
                attempt=attempt,
                citations=len(citations),
            )
            return answer, citations

        except CitationError as exc:
            last_error = exc
            logger.warning(
                "generation.citation_error",
                attempt=attempt,
                error=str(exc),
            )
            question = (
                f"{question}\n\n"
                "IMPORTANT: You MUST cite every claim using [chunk_id] "
                "from the context above. Do not reference any chunk ID "
                "that does not appear in the context."
            )

    raise GenerationError(
        f"Response had invalid citations after {_MAX_RETRIES} attempts: {last_error}"
    )
