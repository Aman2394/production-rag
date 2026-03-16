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


def validate_citations(
    cited_ids: list[str],
    valid_chunk_ids: set[str],
) -> None:
    """Validate that every cited chunk ID exists in the retrieved context.

    Args:
        cited_ids: Pre-extracted list of chunk IDs from the LLM response,
            as returned by ``extract_cited_ids``.
        valid_chunk_ids: Set of chunk IDs from the retrieved context.

    Raises:
        CitationError: If any cited ID is not present in the context.
    """
    invalid = [cid for cid in cited_ids if cid not in valid_chunk_ids]
    if invalid:
        raise CitationError(
            f"Response contains {len(invalid)} invalid citation(s): {invalid}"
        )
