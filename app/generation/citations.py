"""Citation extraction and enforcement for LLM responses."""

import re

from app.exceptions import CitationError

_CITATION_PATTERN = re.compile(r"\[([a-f0-9\-]{36})\]")  # UUID format


def extract_cited_ids(response: str) -> list[str]:
    """Extract all chunk IDs cited in the LLM response.

    Args:
        response: The raw LLM-generated answer string.

    Returns:
        Deduplicated list of chunk ID strings found in the response.
    """
    return list(dict.fromkeys(_CITATION_PATTERN.findall(response)))


def validate_citations(response: str, valid_chunk_ids: set[str]) -> None:
    """Validate that every cited chunk ID exists in the retrieved context.

    Args:
        response: The raw LLM-generated answer string.
        valid_chunk_ids: Set of chunk IDs from the retrieved context.

    Raises:
        CitationError: If the response cites a chunk ID not present in the context.
    """
    cited = extract_cited_ids(response)
    invalid = [cid for cid in cited if cid not in valid_chunk_ids]
    if invalid:
        raise CitationError(
            f"Response contains {len(invalid)} invalid citation(s): {invalid}"
        )
