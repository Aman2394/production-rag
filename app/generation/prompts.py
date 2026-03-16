"""System and user prompt templates for the RAG chain."""

from langchain_core.prompts import ChatPromptTemplate

# ── Base RAG prompt (no history) ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise, grounded Q&A assistant. "
    "Answer ONLY using the provided context. "
    "For every claim you make, cite the source chunk using [chunk_id]. "
    "If the answer is not in the context, reply: "
    '"I don\'t have enough information to answer this." '
    "Do NOT use prior knowledge outside the provided context."
)

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer with citations:"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ]
)

# ── History-aware RAG prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT_WITH_HISTORY = (
    "You are a precise, grounded Q&A assistant with access to conversation history. "
    "Answer ONLY using the provided context. "
    "You may use the conversation history to understand follow-up questions, "
    "but all factual claims must be grounded in the context below. "
    "For every claim you make, cite the source chunk using [chunk_id]. "
    "If the answer is not in the context, reply: "
    '"I don\'t have enough information to answer this." '
    "Do NOT use prior knowledge outside the provided context."
)

USER_PROMPT_WITH_HISTORY_TEMPLATE = """Conversation history:
{history}

Context:
{context}

Question: {question}

Answer with citations:"""

RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_WITH_HISTORY),
        ("human", USER_PROMPT_WITH_HISTORY_TEMPLATE),
    ]
)

# ── Query contextualisation prompt ────────────────────────────────────────────
# Used to reformulate a follow-up question into a standalone retrieval query.

CONTEXTUALIZE_SYSTEM = (
    "Given a conversation history and a follow-up question, "
    "reformulate the follow-up question as a standalone question "
    "that can be understood without the conversation history. "
    "Return ONLY the reformulated question — no explanation, no preamble."
)

CONTEXTUALIZE_USER_TEMPLATE = """Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM),
        ("human", CONTEXTUALIZE_USER_TEMPLATE),
    ]
)


def format_history(messages: list[dict[str, str]]) -> str:
    """Format a list of message dicts into a readable history string.

    Args:
        messages: List of dicts with ``role`` ("human" or "ai") and ``content`` keys,
            ordered oldest → newest.

    Returns:
        Multi-line string with ``Human:`` and ``Assistant:`` prefixes.
    """
    lines = []
    for msg in messages:
        prefix = "Human" if msg["role"] == "human" else "Assistant"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)
