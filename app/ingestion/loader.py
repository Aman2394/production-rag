"""Document loaders for PDF, HTML, and Markdown sources."""

import asyncio
from pathlib import Path
from urllib.parse import urlparse

import structlog
from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from app.exceptions import IngestionError

logger = structlog.get_logger(__name__)

_SUFFIX_LOADERS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
}


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https")


async def _load_file(path: Path) -> list[Document]:
    """Load a single supported file.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        List of LangChain Documents.

    Raises:
        IngestionError: If the file type is unsupported or loading fails.
    """
    suffix = path.suffix.lower()
    loader_cls = _SUFFIX_LOADERS.get(suffix)
    if loader_cls is None:
        raise IngestionError(
            f"Unsupported file type '{suffix}' ({path.name}). "
            f"Supported: {list(_SUFFIX_LOADERS)}"
        )
    try:
        docs: list[Document] = loader_cls(str(path)).load()
        logger.info("loader.file_loaded", file=str(path), pages=len(docs))
        return docs
    except Exception as exc:
        raise IngestionError(f"Failed to load '{path}': {exc}") from exc


async def load_document(source: str) -> list[Document]:
    """Load documents from a file path, directory, or URL.

    - **File**: loads that single document.
    - **Directory**: recursively finds all supported files and loads them
      concurrently via ``asyncio.gather``. Unsupported files are skipped
      with a warning.
    - **URL**: fetches and loads as HTML.

    Args:
        source: File path, directory path, or URL.

    Returns:
        List of LangChain ``Document`` objects.

    Raises:
        IngestionError: If the source doesn't exist, is an unsupported type,
            or loading fails.
    """
    logger.info("loader.start", source=source)

    try:
        if _is_url(source):
            docs = BSHTMLLoader(source).load()
            logger.info("loader.complete", source=source, total=len(docs))
            return docs

        path = Path(source)
        if not path.exists():
            raise IngestionError(f"Path not found: {source}")

        if path.is_dir():
            return await _load_directory(path)

        # Single file
        docs = await _load_file(path)
        logger.info("loader.complete", source=source, total=len(docs))
        return docs

    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Failed to load '{source}': {exc}") from exc


async def _load_directory(directory: Path) -> list[Document]:
    """Recursively load all supported files from a directory concurrently.

    Files with unsupported extensions are skipped with a warning.

    Args:
        directory: Path to the directory to scan.

    Returns:
        Combined list of Documents from all supported files.

    Raises:
        IngestionError: If no supported files are found or all files fail.
    """
    supported_files = [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUFFIX_LOADERS
    ]
    skipped = [
        f.name for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() not in _SUFFIX_LOADERS
    ]

    if skipped:
        logger.warning("loader.skipped_files", count=len(skipped), files=skipped)

    if not supported_files:
        raise IngestionError(
            f"No supported files found in '{directory}'. "
            f"Supported extensions: {list(_SUFFIX_LOADERS)}"
        )

    logger.info("loader.directory_scan", directory=str(directory), files=len(supported_files))

    results = await asyncio.gather(
        *[_load_file(f) for f in supported_files],
        return_exceptions=True,
    )

    all_docs: list[Document] = []
    for f, result in zip(supported_files, results):
        if isinstance(result, Exception):
            logger.error("loader.file_failed", file=str(f), error=str(result))
        else:
            all_docs.extend(result)

    if not all_docs:
        raise IngestionError(f"All files in '{directory}' failed to load.")

    logger.info("loader.complete", directory=str(directory), total=len(all_docs))
    return all_docs
