"""FastAPI application entrypoint."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router
from app.config import get_settings
from app.ingestion.embedder import _get_model as _get_embedder
from app.retrieval.reranker import _get_model as _get_reranker

logger = structlog.get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-load embedding and reranker models before the first request."""
    loop = asyncio.get_event_loop()
    logger.info("startup.warmup.begin")
    await asyncio.gather(
        loop.run_in_executor(None, _get_embedder),
        loop.run_in_executor(None, _get_reranker),
    )
    logger.info("startup.warmup.complete")
    yield


app = FastAPI(
    title="Ask My Docs — Production RAG",
    description="Domain-specific RAG with hybrid retrieval, reranking, and citation enforcement.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
