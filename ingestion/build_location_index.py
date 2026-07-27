import argparse
import sqlite3

import config
from db.connection import get_connection
from utils.embedder import Embedder
from utils.index_builder_helper import build_vector_index

RESOLVER_QUERIES = {
    "locality": "SELECT DISTINCT locality AS value FROM ppd WHERE locality <> ''",
    "town": "SELECT DISTINCT town AS value FROM ppd WHERE town <> ''",
    "district": "SELECT DISTINCT district AS value FROM ppd WHERE district <> ''",
    "county": "SELECT DISTINCT county AS value FROM ppd WHERE county <> ''",
    "region_name": "SELECT DISTINCT region_name AS value FROM hpi WHERE region_name <> ''",
}


def fetch_location_documents(conn: sqlite3.Connection) -> list[dict]:
    documents = []
    for field, sql in RESOLVER_QUERIES.items():
        for row in conn.execute(sql):
            documents.append({"field": field, "value": row["value"]})
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the location-name resolver index")
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument("--out", default=config.LOCATION_INDEX_PATH)
    parser.add_argument("--model-path", default=config.EMBEDDING_MODEL_PATH)
    args = parser.parse_args()

    conn = get_connection(args.db)
    try:
        documents = fetch_location_documents(conn)
    finally:
        conn.close()

    print(f"Building location index over {len(documents)} distinct location values")

    embedder = Embedder(path=args.model_path)
    vector_index = build_vector_index(
        documents, text_field="value", keyword_fields_list=["field"], embedder=embedder
    )
    vector_index.save(args.out)
    print(f"Saved location index to {args.out}")


if __name__ == "__main__":
    main()
