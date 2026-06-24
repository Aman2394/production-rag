"""Request and response Pydantic models for the API layer."""

from pydantic import BaseModel, Field


_NAMESPACE_FIELD = Field(
    default="default",
    pattern=r"^[a-zA-Z0-9_-]{1,64}$",
    description=(
        "Namespace for document isolation. Each namespace is a separate knowledge base "
        "(analogous to a NotebookLM notebook). Alphanumeric, hyphens, underscores only. "
        "Defaults to 'default'."
    ),
)


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    question: str = Field(..., min_length=1, description="The user's question.")
    namespace: str = _NAMESPACE_FIELD
    top_k: int = Field(default=5, gt=0, le=20, description="Number of chunks to retrieve.")
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional session ID for multi-turn conversation memory. "
            "If omitted a new session ID is generated and returned."
        ),
    )


class Citation(BaseModel):
    """A single citation referencing a retrieved chunk."""

    chunk_id: str
    source: str
    page: int | str | None = None


class QueryResponse(BaseModel):
    """Response from POST /query."""

    answer: str
    citations: list[Citation]
    question: str
    session_id: str = Field(description="Session ID — pass this in subsequent requests to continue the conversation.")


class IngestRequest(BaseModel):
    """Payload for POST /ingest (URL or file path)."""

    source: str = Field(..., description="File path or URL to ingest.")
    namespace: str = _NAMESPACE_FIELD


class IngestResponse(BaseModel):
    """Response from POST /ingest."""

    chunks_ingested: int
    source: str
    namespace: str
