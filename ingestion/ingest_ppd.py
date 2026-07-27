import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import config
from db.connection import create_indexes, create_tables, get_connection, set_bulk_load_pragmas
from utils.data_validation_utils import normalize_ppd_dataframe

PPD_COLUMNS = [
    "unique_id", "price_paid", "deed_date", "postcode", "property_type",
    "new_build", "estate_type", "saon", "paon", "street", "locality",
    "town", "district", "county", "ppd_category", "record_status",
]

YEAR_FILE_RE = re.compile(r"^pp-(\d{4})(?:-part(\d+))?\.csv$")


def discover_ppd_files(ppd_dir: str) -> list[Path]:
    """Find year-wise pp-YYYY.csv files in ppd_dir, sorted oldest to newest.

    Some years are split across multiple files, e.g. pp-2010-part1.csv and
    pp-2010-part2.csv (some Land Registry years are too large for a single
    download) - those sort after each other by part number, within the year.
    A plain pp-YYYY.csv (no -partN) is treated as part 0.

    Only matches this strict naming (not e.g. renamed backups like
    pp-complete.csv_full) so stray files in the directory are ignored.
    """
    matches = []
    for path in Path(ppd_dir).iterdir():
        m = YEAR_FILE_RE.match(path.name)
        if m:
            year = int(m.group(1))
            part = int(m.group(2)) if m.group(2) else 0
            matches.append((year, part, path))
    matches.sort(key=lambda triple: (triple[0], triple[1]))
    return [path for _, _, path in matches]


def _insert_sql(or_ignore: bool) -> str:
    verb = "INSERT OR IGNORE" if or_ignore else "INSERT"
    return f"""
        {verb} INTO ppd ({", ".join(PPD_COLUMNS)})
        VALUES ({", ".join("?" for _ in PPD_COLUMNS)})
    """


def _insert_chunk(conn: sqlite3.Connection, df: pd.DataFrame, or_ignore: bool) -> int:
    df = normalize_ppd_dataframe(df)
    rows = list(df[PPD_COLUMNS].itertuples(index=False, name=None))
    with conn:
        conn.executemany(_insert_sql(or_ignore), rows)

    # SQLite's automatic WAL checkpoint can fall behind under a sustained
    # heavy-write load like this, letting the WAL file balloon to gigabytes
    # (observed: 2.5GB+ mid-run). A bigger WAL means every subsequent commit
    # has to work through a bigger wal-index, which is what caused this
    # ingest to slow from ~14k rows/s to ~5k rows/s partway through. Forcing
    # a cheap PASSIVE checkpoint after every chunk keeps the WAL bounded
    # instead of relying on the (here, ineffective) automatic scheduler.
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    return len(rows)


def ingest_ppd(
    ppd_dir: str | None = None,
    csv_paths: list[str] | None = None,
    db_path: str = config.DB_PATH,
    limit: int = 0,
    chunksize: int = config.PPD_INGEST_CHUNKSIZE,
    append: bool = False,
) -> int:
    if csv_paths:
        files = [Path(p) for p in csv_paths]
    else:
        files = discover_ppd_files(ppd_dir or config.PPD_DIR)

    if not files:
        raise FileNotFoundError(f"No pp-YYYY.csv files found in {ppd_dir or config.PPD_DIR}")

    conn = get_connection(db_path)
    try:
        create_tables(conn)
        set_bulk_load_pragmas(conn)

        if not append:
            conn.execute("DELETE FROM ppd")
            conn.commit()

        inserted = 0
        with tqdm(desc="Ingesting PPD rows", unit="rows") as progress:
            for file_path in files:
                progress.set_description(f"Ingesting PPD rows [{file_path.name}]")
                reader = pd.read_csv(
                    file_path,
                    dtype=str,
                    keep_default_na=False,
                    chunksize=chunksize,
                )

                for chunk in reader:
                    if limit and inserted + len(chunk) > limit:
                        chunk = chunk.iloc[: limit - inserted]

                    inserted += _insert_chunk(conn, chunk, or_ignore=append)
                    progress.update(len(chunk))

                    if limit and inserted >= limit:
                        break

                if limit and inserted >= limit:
                    break

        # Uniqueness on unique_id is enforced here, in one bulk pass, rather
        # than checked on every insert above (see schema_tables.sql) - by far
        # the biggest lever on ingest throughput at this scale.
        create_indexes(conn)
        conn.execute("ANALYZE ppd")
        conn.commit()
        return conn.execute("SELECT COUNT(*) FROM ppd").fetchone()[0]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest year-wise Price Paid Data CSVs into SQLite")
    parser.add_argument(
        "--ppd-dir",
        default=config.PPD_DIR,
        help="Directory to scan for pp-YYYY.csv files",
    )
    parser.add_argument(
        "--csv",
        action="append",
        dest="csv_paths",
        help="Ingest only this file instead of scanning --ppd-dir (repeatable)",
    )
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=config.PPD_INGEST_DEFAULT_LIMIT,
        help="Max rows to insert across all files combined (0 = unbounded)",
    )
    parser.add_argument("--chunksize", type=int, default=config.PPD_INGEST_CHUNKSIZE)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing ppd rows instead of truncating first",
    )
    args = parser.parse_args()

    count = ingest_ppd(
        ppd_dir=args.ppd_dir,
        csv_paths=args.csv_paths,
        db_path=args.db,
        limit=args.limit,
        chunksize=args.chunksize,
        append=args.append,
    )
    print(f"ppd table now has {count} rows in {args.db}")


if __name__ == "__main__":
    main()
