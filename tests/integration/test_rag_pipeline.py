"""Integration tests for the end-to-end RAG pipeline.

Requires a running Qdrant instance (docker-compose up -d qdrant).
"""

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_placeholder() -> None:
    """Placeholder — replace with real pipeline integration tests."""
    # TODO: spin up Qdrant, ingest a sample doc, run a query, assert citations
    assert True
