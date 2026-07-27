"""Search-quality metrics, following the llm-zoomcamp evaluation methodology
(https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/04-search-evaluation.md).

`search_function` is any callable of (question: str, field: str) -> list[dict],
where each returned dict has at least a "value" key - this matches
LocationResolver.resolve's return shape, so no changes are needed to the
existing resolver/index code to evaluate it.
"""

from tqdm.auto import tqdm


def compute_relevance(ground_truth_record: dict, search_function) -> list[int]:
    results = search_function(ground_truth_record["question"], ground_truth_record["field"])
    return [int(r["value"] == ground_truth_record["value"]) for r in results]


def compute_relevance_total(ground_truth: list[dict], search_function) -> list[list[int]]:
    relevance_total = []
    for record in tqdm(ground_truth, desc="Evaluating"):
        relevance_total.append(compute_relevance(record, search_function))
    return relevance_total


def hit_rate(relevance_total: list[list[int]]) -> float:
    cnt = sum(1 for line in relevance_total if 1 in line)
    return cnt / len(relevance_total)


def mrr(relevance_total: list[list[int]]) -> float:
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth: list[dict], search_function) -> dict:
    relevance_total = compute_relevance_total(ground_truth, search_function)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }
