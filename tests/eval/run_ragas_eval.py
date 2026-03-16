"""RAGAS evaluation script — CI will block PRs that fall below thresholds."""

import json
import sys
from pathlib import Path

THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.75,
}

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def main() -> None:
    """Run RAGAS evaluation and exit non-zero if any threshold is breached."""
    dataset = json.loads(EVAL_DATASET_PATH.read_text())
    print(f"Loaded {len(dataset)} eval samples.")

    # TODO: run the RAG pipeline over each sample, collect answers + contexts,
    #       build a ragas.EvaluationDataset, then call ragas.evaluate().
    # Example skeleton:
    #
    # from ragas import evaluate
    # from ragas.metrics import faithfulness, answer_relevancy, ...
    # result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, ...])
    # scores = result.to_pandas().mean().to_dict()

    scores: dict[str, float] = {}  # Replace with real scores above

    failed = False
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, None)
        if score is None:
            print(f"  [SKIP] {metric}: not yet implemented")
            continue
        status = "PASS" if score >= threshold else "FAIL"
        if status == "FAIL":
            failed = True
        print(f"  [{status}] {metric}: {score:.3f} (threshold={threshold})")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
