"""POST /ingest — load, chunk, embed, and store documents."""

import structlog
from fastapi import APIRouter, HTTPException

from app.api.schemas import IngestRequest, IngestResponse
from app.exceptions import IngestionError
from app.ingestion.pipeline import run_ingestion

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest a document from a file path or URL into the vector store.

    Args:
        request: The ingest payload containing the source path or URL.

    Returns:
        IngestResponse with the number of chunks ingested.

    Raises:
        HTTPException: On ingestion failure.
    """
    logger.info("ingest.received", source=request.source)
    try:
        chunks_ingested = await run_ingestion(request.source)
        return IngestResponse(chunks_ingested=chunks_ingested, source=request.source)
    except IngestionError as exc:
        logger.error("ingest.failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
