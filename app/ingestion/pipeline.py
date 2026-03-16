"""Async ingestion orchestration: load → chunk → embed → store."""

import asyncio
import uuid
from datetime import datetime, timezone

import structlog
from langchain_core.documents import Document

from app.exceptions import IngestionError
from app.ingestion.chunker import build_splitter
from app.ingestion.embedder import embed_texts
from app.ingestion.loader import load_document
from app.retrieval.bm25_store import add_chunks
from app.retrieval.vector_store import upsert_chunks

logger = structlog.get_logger(__name__)


def _build_chunks(docs: list[Document], source: str) -> list[dict]:
    """Split LangChain Documents into chunks with required metadata.

    Args:
        docs: Raw documents returned by the loader.
        source: Original source path or URL (used as metadata).

    Returns:
        List of chunk dicts with ``chunk_id``, ``content``, and ``metadata``.
    """
    splitter = build_splitter()
    ingested_at = datetime.now(timezone.utc).isoformat()
    chunks: list[dict] = []

    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        page = doc.metadata.get("page", doc.metadata.get("section", None))

        for text in splits:
            if not text.strip():
                continue
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "content": text,
                    "metadata": {
                        "source": source,
                        "page": page,
                        "ingested_at": ingested_at,
                    },
                }
            )

    return chunks


async def run_ingestion(source: str) -> int:
    """Run the full ingestion pipeline for a single source.

    Loads the document, splits it into chunks, embeds all chunks in
    parallel batches, then upserts to Qdrant and the BM25 index concurrently.

    Args:
        source: File path or URL to ingest.

    Returns:
        Number of chunks successfully ingested.

    Raises:
        IngestionError: If any stage of the pipeline fails.
    """
    logger.info("ingestion.start", source=source)
    try:
        # 1. Load
        docs = await load_document(source)

        # 2. Chunk
        chunks = _build_chunks(docs, source)
        if not chunks:
            logger.warning("ingestion.no_chunks", source=source)
            return 0

        logger.info("ingestion.chunks_created", source=source, count=len(chunks))

        # 3. Embed
        texts = [c["content"] for c in chunks]
        embeddings = await embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        # 4. Store — upsert to Qdrant and BM25 concurrently
        await asyncio.gather(
            upsert_chunks(chunks),
            add_chunks(chunks),
        )

        logger.info("ingestion.complete", source=source, chunks=len(chunks))
        return len(chunks)

    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Ingestion failed for '{source}': {exc}") from exc
