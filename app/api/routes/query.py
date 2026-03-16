"""POST /query — answer a question using the RAG pipeline with conversation memory."""

import uuid

import structlog
from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse
from app.exceptions import GenerationError, RetrievalError
from app.generation.chain import contextualize_question, generate
from app.memory.manager import get_history, save_turn
from app.retrieval.pipeline import retrieve

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a user question using hybrid retrieval + reranking + LLM generation.

    Supports multi-turn conversations via ``session_id``. On each request:
    1. Loads conversation history from Redis for the session.
    2. Contextualises the question (if history exists) for better retrieval.
    3. Runs hybrid retrieval (BM25 + dense) → RRF → BGE reranking.
    4. Generates a grounded, cited answer using Claude.
    5. Saves the turn to Redis (short-term) and optionally PostgreSQL (long-term).

    Args:
        request: The query payload with question, optional top_k and session_id.

    Returns:
        QueryResponse with the grounded answer, citations, and session_id.

    Raises:
        HTTPException 503: If no documents have been ingested yet.
        HTTPException 500: On retrieval or generation failure.
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("query.received", question=request.question, session_id=session_id)

    try:
        # 1. Load short-term conversation history
        history = await get_history(session_id)

        # 2. Contextualise the question for retrieval if there's prior history
        retrieval_query = await contextualize_question(request.question, history)

        # 3. Retrieve relevant chunks using the standalone query
        chunks = await retrieve(retrieval_query, top_n=request.top_k)

        # 4. Generate grounded answer with history context
        answer, citations = await generate(
            request.question,
            chunks,
            history=history or None,
        )

        # 5. Save turn to memory layers (non-blocking failures handled inside)
        citations_as_dicts = [c.model_dump() for c in citations]
        await save_turn(session_id, request.question, answer, citations_as_dicts)

        logger.info(
            "query.complete",
            session_id=session_id,
            citations=len(citations),
        )
        return QueryResponse(
            answer=answer,
            citations=citations,
            question=request.question,
            session_id=session_id,
        )

    except RetrievalError as exc:
        logger.error("query.retrieval_failed", error=str(exc))
        if "empty" in str(exc).lower():
            raise HTTPException(
                status_code=503,
                detail="No documents ingested yet. POST to /ingest first.",
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except GenerationError as exc:
        logger.error("query.generation_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
