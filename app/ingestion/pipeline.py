"""Async ingestion orchestration: load → chunk → embed → store."""

import structlog

from app.exceptions import IngestionError
from app.ingestion.chunker import build_splitter
from app.ingestion.embedder import embed_texts
from app.ingestion.loader import load_document

logger = structlog.get_logger(__name__)


async def run_ingestion(source: str) -> int:
    """Run the full ingestion pipeline for a single source.

    Args:
        source: File path or URL to ingest.

    Returns:
        Number of chunks successfully ingested.

    Raises:
        IngestionError: If any stage of the pipeline fails.
    """
    logger.info("ingestion.start", source=source)
    try:
        documents = await load_document(source)
        splitter = build_splitter()
        # TODO: split, embed, and upsert to Qdrant + BM25
        logger.info("ingestion.complete", source=source, chunks=0)
        return 0
    except Exception as exc:
        raise IngestionError(f"Ingestion failed for {source}: {exc}") from exc
