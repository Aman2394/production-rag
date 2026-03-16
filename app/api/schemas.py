"""Request and response Pydantic models for the API layer."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    question: str = Field(..., min_length=1, description="The user's question.")
    top_k: int = Field(default=5, gt=0, le=20, description="Number of chunks to retrieve.")


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


class IngestRequest(BaseModel):
    """Payload for POST /ingest (URL or file path)."""

    source: str = Field(..., description="File path or URL to ingest.")


class IngestResponse(BaseModel):
    """Response from POST /ingest."""

    chunks_ingested: int
    source: str
