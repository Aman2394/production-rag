"""System and user prompt templates for the RAG chain."""

from langchain_core.prompts import ChatPromptTemplate

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
