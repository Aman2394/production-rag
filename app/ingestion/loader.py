"""Document loaders for PDF, HTML, and Markdown sources."""

from app.exceptions import IngestionError


async def load_document(source: str) -> list[dict[str, str]]:
    """Load a document from a file path or URL.

    Args:
        source: File path or URL pointing to a PDF, HTML, or Markdown document.

    Returns:
        List of dicts with ``content`` and ``metadata`` keys.

    Raises:
        IngestionError: If the source cannot be loaded or the format is unsupported.
    """
    # TODO: implement loaders using LangChain document loaders
    raise IngestionError(f"Loader not yet implemented for source: {source}")
