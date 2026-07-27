"""Evaluate the location vector search (rag/resolver.py's LocationResolver)
against a ground-truth CSV using hit_rate and MRR, following the llm-zoomcamp
evaluation methodology:
https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/04-search-evaluation.md
https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/05-search-metrics.md

Usage:
    uv run python -m evaluation.generate_ground_truth   # one-time, costs a few LLM calls
    uv run python -m evaluation.evaluate_location_search
"""

import argparse
import csv

import config
from evaluation.metrics import evaluate
from rag.resolver import LocationResolver
from utils.embedder import Embedder


def load_ground_truth(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the location vector search")
    parser.add_argument(
        "--ground-truth", default=str(config.BASE_DIR / "data" / "ground_truth_location.csv")
    )
    parser.add_argument("--index", default=config.LOCATION_INDEX_PATH)
    parser.add_argument("--model-path", default=config.EMBEDDING_MODEL_PATH)
    parser.add_argument("--num-results", type=int, default=10)
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)

    embedder = Embedder(path=args.model_path)
    resolver = LocationResolver.load(args.index, embedder)

    def search_function(question: str, field: str) -> list[dict]:
        return resolver.resolve(question, field=field, num_results=args.num_results)

    metrics = evaluate(ground_truth, search_function)

    print(f"Evaluated {len(ground_truth)} ground-truth queries against the location index")
    print(f"hit_rate@{args.num_results}: {metrics['hit_rate']:.3f}")
    print(f"mrr@{args.num_results}:      {metrics['mrr']:.3f}")


if __name__ == "__main__":
    main()
