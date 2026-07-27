"""Generate a ground-truth CSV (question -> correct field/value) for
evaluating the location vector search, following the llm-zoomcamp approach of
using an LLM to synthesize realistic user queries for known documents:
https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-evaluation/lessons/02-ground-truth.md

Samples a subset of the distinct locality/town/district/county/region_name
values (the same documents ingestion/build_location_index.py embeds) and asks
the LLM to emulate how a UK home buyer might actually type each one into a
search box - typos, abbreviations, partial/colloquial forms - rather than
the exact canonical spelling.
"""

import argparse
import csv
import random

from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

import config
from db.connection import get_connection
from ingestion.build_location_index import fetch_location_documents
from utils.evaluation_utils import calc_price, llm_structured

GENERATION_INSTRUCTIONS = """
You emulate a UK home buyer searching for a location by typing it into a
search box. Given the canonical name of a UK locality/town/district/county/
region, generate 5 different ways a real user might actually type it when
searching for it - natural typos, missing or extra spaces, dropped
punctuation, common misspellings, abbreviations, or colloquial/partial
forms. Do not just repeat the exact canonical value verbatim in any of the
5. Keep each variant short, like a real search query, not a full sentence.
""".strip()


class LocationQueries(BaseModel):
    queries: list[str]


def sample_documents(documents: list[dict], sample_size_per_field: int, seed: int) -> list[dict]:
    by_field: dict[str, list[dict]] = {}
    for doc in documents:
        by_field.setdefault(doc["field"], []).append(doc)

    rng = random.Random(seed)
    sampled = []
    for field_docs in by_field.values():
        sampled.extend(rng.sample(field_docs, min(sample_size_per_field, len(field_docs))))
    return sampled


def build_ground_truth(
    documents: list[dict], client: OpenAI, model: str
) -> tuple[list[dict], float]:
    records = []
    total_cost = 0.0

    for doc in tqdm(documents, desc="Generating ground-truth queries"):
        user_prompt = f"Canonical {doc['field']} name: {doc['value']}"
        result, usage = llm_structured(
            client, GENERATION_INSTRUCTIONS, user_prompt, LocationQueries, model=model
        )
        total_cost += calc_price(usage)["total_cost"]

        for question in result.queries:
            records.append({"question": question, "field": doc["field"], "value": doc["value"]})

    return records, total_cost


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth questions for evaluating the location vector search"
    )
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument(
        "--out", default=str(config.BASE_DIR / "data" / "ground_truth_location.csv")
    )
    parser.add_argument(
        "--sample-size-per-field",
        type=int,
        default=20,
        help="How many distinct values to sample per field (locality/town/district/county/region_name)",
    )
    parser.add_argument("--model", default=config.EXTRACTION_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conn = get_connection(args.db)
    try:
        documents = fetch_location_documents(conn)
    finally:
        conn.close()

    sampled = sample_documents(documents, args.sample_size_per_field, args.seed)
    print(f"Sampled {len(sampled)} location values to generate queries for")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    records, total_cost = build_ground_truth(sampled, client, args.model)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "field", "value"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Generation cost: ${total_cost:.4f}")
    print(f"Wrote {len(records)} ground-truth records to {args.out}")


if __name__ == "__main__":
    main()
