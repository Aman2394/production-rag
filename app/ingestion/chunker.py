"""Text chunking using RecursiveCharacterTextSplitter."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

_settings = get_settings()

SEPARATORS = ["\n\n", "\n", ". ", " "]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Build a configured text splitter.

    Args:
        chunk_size: Override the default chunk size from settings.
        chunk_overlap: Override the default chunk overlap from settings.

    Returns:
        A configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or _settings.chunk_size,
        chunk_overlap=chunk_overlap or _settings.chunk_overlap,
        separators=SEPARATORS,
    )
