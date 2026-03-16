"""LangChain LCEL RAG chain: context formatting → prompt → LLM → citation validation."""

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import get_settings
from app.generation.citations import validate_citations
from app.generation.prompts import RAG_PROMPT

logger = structlog.get_logger(__name__)
_settings = get_settings()


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into the context block for the prompt.

    Args:
        chunks: List of chunk dicts containing ``chunk_id`` and ``content``.

    Returns:
        Formatted context string with chunk IDs as headers.
    """
    parts = [f"[{c['chunk_id']}]\n{c['content']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def build_rag_chain() -> object:
    """Build and return the LCEL RAG chain.

    Returns:
        A compiled LangChain runnable that accepts ``{"question": str, "chunks": list}``
        and returns the grounded answer string.
    """
    llm = ChatAnthropic(
        model=_settings.anthropic_model,
        api_key=_settings.anthropic_api_key.get_secret_value(),
    )

    chain = (
        {
            "context": lambda x: _format_context(x["chunks"]),
            "question": RunnablePassthrough() | (lambda x: x["question"]),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
