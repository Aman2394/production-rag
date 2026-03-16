"""Unit tests for the document loader."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.exceptions import IngestionError
from app.ingestion.loader import load_document, _load_directory


@pytest.mark.asyncio
async def test_load_document_raises_for_missing_file() -> None:
    with pytest.raises(IngestionError, match="Path not found"):
        await load_document("/nonexistent/path/file.pdf")


@pytest.mark.asyncio
async def test_load_document_raises_for_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "doc.xyz"
    f.write_text("content")
    with pytest.raises(IngestionError, match="Unsupported file type"):
        await load_document(str(f))


@pytest.mark.asyncio
async def test_load_document_directory_loads_all_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Hello\nThis is document A.")
    (tmp_path / "b.md").write_text("# World\nThis is document B.")
    (tmp_path / "ignored.txt").write_text("skip me")

    from langchain_core.documents import Document

    mock_docs = [Document(page_content="mocked content", metadata={})]

    with patch("app.ingestion.loader._load_file", AsyncMock(return_value=mock_docs)):
        docs = await load_document(str(tmp_path))

    # Two .md files → 2 × mock_docs
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_load_directory_raises_when_no_supported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not supported")
    with pytest.raises(IngestionError, match="No supported files found"):
        await _load_directory(tmp_path)


@pytest.mark.asyncio
async def test_load_directory_skips_failed_files_but_returns_rest(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("# Good doc")
    (tmp_path / "bad.md").write_text("# Bad doc")

    from langchain_core.documents import Document
    good_docs = [Document(page_content="good", metadata={})]

    call_count = 0

    async def mock_load(path: Path) -> list[Document]:
        nonlocal call_count
        call_count += 1
        if "bad" in path.name:
            raise IngestionError("simulated failure")
        return good_docs

    with patch("app.ingestion.loader._load_file", side_effect=mock_load):
        docs = await _load_directory(tmp_path)

    assert len(docs) == 1
    assert call_count == 2
