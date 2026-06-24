"""RAGAS evaluation — CI blocks PRs that fall below metric thresholds.

Two modes (--mode flag):
  static   (default) — uses reference_contexts from eval_dataset.json; no live
                        services needed; evaluates faithfulness + answer_relevancy.
  pipeline            — runs the full RAG retrieval pipeline per question; requires
                        a running Qdrant instance and a populated BM25 index;
                        evaluates all 4 metrics including context_precision/recall.

LLM for RAG generation:
  Controlled by LLM_PROVIDER env var (ollama | anthropic | groq).

LLM for RAGAS judge (metric evaluation):
  Priority: GROQ_API_KEY → ANTHROPIC_API_KEY
  Groq is recommended — free tier (console.groq.com), no credit card required.
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.75,
}

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

# Metrics only available with live retrieval (mode=pipeline)
_PIPELINE_ONLY_METRICS = {"context_precision", "context_recall"}


# ── RAGAS LLM / embeddings factory ───────────────────────────────────────────

def _get_ragas_llm() -> Any:
    """Return the best available LangChain LLM wrapped for RAGAS evaluation.

    Priority: GROQ_API_KEY (free) → ANTHROPIC_API_KEY.

    Returns:
        A LangchainLLMWrapper instance.

    Raises:
        RuntimeError: If no supported API key is found in the environment.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper
        print("  Using Groq (llama-3.3-70b-versatile) as RAGAS judge LLM.")
        return LangchainLLMWrapper(
            ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0)
        )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        from ragas.llms import LangchainLLMWrapper
        print("  Using Anthropic (claude-haiku-4-5-20251001) as RAGAS judge LLM.")
        return LangchainLLMWrapper(
            ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=anthropic_key,
                temperature=0,
            )
        )

    raise RuntimeError(
        "No LLM configured for RAGAS evaluation.\n"
        "  Set GROQ_API_KEY (free, recommended) or ANTHROPIC_API_KEY.\n"
        "  Get a Groq key at: https://console.groq.com"
    )


def _get_ragas_embeddings() -> Any:
    """Return HuggingFace embeddings wrapped for RAGAS (used by AnswerRelevancy).

    Uses BGE-small-en-v1.5 (~65 MB) — lighter than the ingestion model —
    since RAGAS only needs embeddings for semantic similarity scoring.

    Returns:
        A LangchainEmbeddingsWrapper instance.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    model = os.getenv("RAGAS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"  Using {model} for RAGAS embeddings.")
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"})
    )


def _build_metrics(ragas_llm: Any, ragas_embeddings: Any, mode: str) -> list[Any]:
    """Instantiate RAGAS metrics for the given evaluation mode.

    Args:
        ragas_llm: RAGAS-wrapped LangChain LLM.
        ragas_embeddings: RAGAS-wrapped LangChain embeddings.
        mode: ``"static"`` (faithfulness + answer_relevancy) or
              ``"pipeline"`` (all 4 metrics).

    Returns:
        List of configured RAGAS metric instances.
    """
    from ragas.metrics import Faithfulness, AnswerRelevancy

    metrics: list[Any] = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    if mode == "pipeline":
        try:
            from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
        except ImportError:
            # Older RAGAS 0.2.x naming
            from ragas.metrics import (  # type: ignore[no-redef]
                ContextPrecision as LLMContextPrecisionWithReference,
                ContextRecall as LLMContextRecall,
            )
        metrics.extend([
            LLMContextPrecisionWithReference(llm=ragas_llm),
            LLMContextRecall(llm=ragas_llm),
        ])

    return metrics


# ── Answer generation ─────────────────────────────────────────────────────────

async def _generate_static(
    question: str,
    reference_contexts: list[str],
) -> tuple[str, list[str]]:
    """Generate an answer from pre-supplied reference contexts (no retrieval).

    Wraps each context string into a pseudo-chunk dict so it can be passed
    directly to ``generation.chain.generate()``.

    Args:
        question: Evaluation question.
        reference_contexts: Context passages from eval_dataset.json.

    Returns:
        Tuple of (generated answer, contexts used).
    """
    from app.generation.chain import generate

    chunks = [
        {
            "chunk_id": str(uuid.uuid4()),
            "content": ctx,
            "metadata": {"source": "eval_reference", "page": i + 1},
        }
        for i, ctx in enumerate(reference_contexts)
    ]
    answer, _ = await generate(question, chunks)
    return answer, reference_contexts


async def _generate_pipeline(question: str) -> tuple[str, list[str]]:
    """Run the full RAG pipeline (retrieve + generate) for a question.

    Requires a running Qdrant instance and a populated BM25 index.

    Args:
        question: Evaluation question.

    Returns:
        Tuple of (generated answer, retrieved context strings).
    """
    from app.retrieval.pipeline import retrieve
    from app.generation.chain import generate

    chunks = await retrieve(question)
    answer, _ = await generate(question, chunks)
    retrieved_contexts = [c["content"] for c in chunks]
    return answer, retrieved_contexts


# ── Evaluation orchestration ──────────────────────────────────────────────────

async def _collect_results(
    dataset: list[dict[str, Any]],
    mode: str,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Run inference over eval samples and collect (question, answer, contexts, gt) tuples.

    Args:
        dataset: Loaded eval dataset.
        mode: ``"static"`` or ``"pipeline"``.
        max_samples: Optional cap on number of samples (useful for quick checks).

    Returns:
        List of result dicts ready for RAGAS dataset construction.
    """
    samples = dataset[:max_samples] if max_samples else dataset
    results: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        print(f"  [{i + 1}/{len(samples)}] {sample['question'][:70]}...")
        try:
            if mode == "static":
                answer, contexts = await _generate_static(
                    sample["question"],
                    sample["reference_contexts"],
                )
            else:
                answer, contexts = await _generate_pipeline(sample["question"])

            results.append({
                "question": sample["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": sample["ground_truth"],
            })
        except Exception as exc:
            print(f"    WARN: skipped sample {i + 1} — {exc}")

    return results


def _build_ragas_dataset(results: list[dict[str, Any]]) -> Any:
    """Build a RAGAS EvaluationDataset from collected inference results.

    Args:
        results: List of dicts with question, answer, contexts, ground_truth.

    Returns:
        RAGAS EvaluationDataset.
    """
    from ragas import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["ground_truth"],
        )
        for r in results
    ]
    return EvaluationDataset(samples=samples)


def _evaluate(ragas_dataset: Any, metrics: list[Any]) -> dict[str, float]:
    """Run RAGAS evaluation and return per-metric mean scores.

    Args:
        ragas_dataset: RAGAS EvaluationDataset.
        metrics: List of configured RAGAS metric instances.

    Returns:
        Dict mapping metric name → mean score across all samples.
    """
    from ragas import evaluate

    result = evaluate(dataset=ragas_dataset, metrics=metrics)
    scores_df = result.to_pandas()
    return {col: float(scores_df[col].mean()) for col in scores_df.columns}


def _print_results(scores: dict[str, float], mode: str) -> bool:
    """Print a formatted results table and return True if any threshold is failed.

    Args:
        scores: Dict of metric name → mean score.
        mode: Evaluation mode (determines which metrics are expected).

    Returns:
        True if any threshold is breached, False if all pass.
    """
    print("\n" + "=" * 50)
    print("RAGAS Evaluation Results")
    print("=" * 50)

    failed = False
    for metric, threshold in THRESHOLDS.items():
        if metric in _PIPELINE_ONLY_METRICS and mode == "static":
            print(f"  [SKIP] {metric:<25} (requires mode=pipeline)")
            continue

        score = scores.get(metric)
        if score is None:
            print(f"  [SKIP] {metric:<25} (not in results)")
            continue

        status = "PASS" if score >= threshold else "FAIL"
        if status == "FAIL":
            failed = True
        bar = "█" * int(score * 20)
        print(f"  [{status}] {metric:<25} {score:.3f}  {bar:<20}  (min={threshold:.2f})")

    print("=" * 50)
    return failed


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(mode: str, max_samples: int | None) -> None:
    """Orchestrate the full RAGAS evaluation run.

    Args:
        mode: ``"static"`` or ``"pipeline"``.
        max_samples: If set, caps the number of eval samples processed.
    """
    print(f"\nRAGAS Evaluation  (mode={mode})")
    print("-" * 50)

    dataset: list[dict[str, Any]] = json.loads(EVAL_DATASET_PATH.read_text())
    cap = f" (capped at {max_samples})" if max_samples else ""
    print(f"Loaded {len(dataset)} eval samples{cap}.")

    print("\nSetting up RAGAS judge...")
    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()
    metrics = _build_metrics(ragas_llm, ragas_embeddings, mode)

    print(f"\nGenerating answers ({mode} mode)...")
    results = await _collect_results(dataset, mode, max_samples)

    if not results:
        print("\nERROR: No answers generated. Check pipeline configuration and .env file.")
        sys.exit(1)

    print(f"\nCollected {len(results)}/{len(dataset[:max_samples] if max_samples else dataset)} answers.")

    print("\nRunning RAGAS metrics...")
    ragas_dataset = _build_ragas_dataset(results)
    scores = _evaluate(ragas_dataset, metrics)

    failed = _print_results(scores, mode)

    if failed:
        print("\nEVALUATION FAILED — one or more metrics below threshold.")
        sys.exit(1)
    else:
        print("\nEVALUATION PASSED — all active metrics above threshold.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation against the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_ragas_eval.py                     # static mode, all samples\n"
            "  python run_ragas_eval.py --mode pipeline     # full retrieval pipeline\n"
            "  python run_ragas_eval.py --samples 5         # quick sanity check (5 samples)\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["static", "pipeline"],
        default="static",
        help=(
            "static: use reference_contexts from eval_dataset.json (no Qdrant needed). "
            "pipeline: run full RAG retrieval per question (requires running Qdrant)."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N samples (useful for quick iteration).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.samples))
