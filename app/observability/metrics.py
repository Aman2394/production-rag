"""Latency and cost-per-request tracking."""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog

logger = structlog.get_logger(__name__)

# Approximate cost per token (USD) — update as pricing changes
_COST_PER_INPUT_TOKEN = 3e-6   # claude-sonnet-4-6 input
_COST_PER_OUTPUT_TOKEN = 15e-6  # claude-sonnet-4-6 output


@asynccontextmanager
async def track_request(operation: str) -> AsyncGenerator[None, None]:
    """Async context manager that logs latency for a RAG operation.

    Args:
        operation: Human-readable name for the operation being timed.

    Yields:
        None — use as ``async with track_request("retrieval"): ...``
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("metrics.latency", operation=operation, latency_ms=round(elapsed_ms, 2))


def log_token_cost(input_tokens: int, output_tokens: int) -> float:
    """Log and return the estimated USD cost for a single LLM call.

    Args:
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.

    Returns:
        Estimated cost in USD.
    """
    cost = input_tokens * _COST_PER_INPUT_TOKEN + output_tokens * _COST_PER_OUTPUT_TOKEN
    logger.info(
        "metrics.cost",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
    )
    return cost
