"""POST /query and POST /query/stream — RAG pipeline with conversation memory."""

import json
import uuid
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse
from app.exceptions import GenerationError, RetrievalError
from app.generation.chain import contextualize_question, generate, stream_generate
from app.memory.manager import get_history, save_turn
from app.observability.metrics import track_request
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
        async with track_request("total"):
            # 1. Load short-term conversation history
            async with track_request("memory.load"):
                history = await get_history(session_id)

            # 2. Contextualise the question for retrieval if there's prior history
            async with track_request("contextualize"):
                retrieval_query = await contextualize_question(request.question, history)

            # 3. Retrieve relevant chunks using the standalone query
            async with track_request("retrieval"):
                chunks = await retrieve(retrieval_query, namespace=request.namespace, top_n=request.top_k)

            # 4. Generate grounded answer with history context
            async with track_request("generation"):
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


@router.post("/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Stream a RAG answer token-by-token via Server-Sent Events (SSE).

    Runs history + contextualise + retrieve synchronously first (fast), then
    streams LLM generation tokens as they arrive. Saves the completed turn to
    memory after the last token is sent.

    SSE event format — one JSON object per ``data:`` line::

        {"type": "token",  "content": "<token>"}
        {"type": "done",   "session_id": "...", "answer": "...", "citations": [...]}
        {"type": "error",  "detail": "<message>"}

    Connect from the browser with the ``EventSource`` API or any SSE client.
    ``curl`` example::

        curl -N -X POST http://localhost:8000/query/stream \\
             -H "Content-Type: application/json" \\
             -d '{"question": "What is RRF?"}'

    Args:
        request: Same ``QueryRequest`` payload as ``POST /query``.

    Returns:
        A ``text/event-stream`` StreamingResponse.
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("query_stream.received", question=request.question, session_id=session_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        # ── Non-streaming pre-steps (history → contextualise → retrieve) ─────
        try:
            async with track_request("memory.load"):
                history = await get_history(session_id)

            async with track_request("contextualize"):
                retrieval_query = await contextualize_question(request.question, history)

            async with track_request("retrieval"):
                chunks = await retrieve(retrieval_query, namespace=request.namespace, top_n=request.top_k)

        except RetrievalError as exc:
            detail = (
                "No documents ingested yet. POST to /ingest first."
                if "empty" in str(exc).lower()
                else str(exc)
            )
            yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"
            return
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            return

        # ── Streaming generation ──────────────────────────────────────────────
        answer = ""
        citations_data: list[dict] = []

        async with track_request("generation.stream"):
            async for event in stream_generate(request.question, chunks, history or None):
                if event["type"] == "done":
                    answer = event["answer"]
                    citations_data = event["citations"]
                    event["session_id"] = session_id
                yield f"data: {json.dumps(event)}\n\n"

        # ── Persist the completed turn ────────────────────────────────────────
        if answer:
            await save_turn(session_id, request.question, answer, citations_data)
            logger.info(
                "query_stream.complete",
                session_id=session_id,
                citations=len(citations_data),
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for real-time delivery
        },
    )
