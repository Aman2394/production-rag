"""POST /query — answer a question using the RAG pipeline."""

import structlog
from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse
from app.exceptions import GenerationError, RetrievalError

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a user question using hybrid retrieval + reranking + LLM generation.

    Args:
        request: The query payload containing the user's question.

    Returns:
        QueryResponse with the answer and inline citations.

    Raises:
        HTTPException: On retrieval or generation failure.
    """
    logger.info("query.received", question=request.question)
    try:
        # TODO: wire up retrieval.pipeline and generation.chain
        raise NotImplementedError("RAG pipeline not yet implemented.")
    except (RetrievalError, GenerationError) as exc:
        logger.error("query.failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
