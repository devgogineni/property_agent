import sqlite3
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent


def get_connection(db_path: str | Path, check_same_thread: bool = True) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    # Without this, a second connection trying to write while another holds
    # the single writer lock (e.g. running ingest_hpi while ingest_ppd's bulk
    # load is still in progress) fails immediately with "database is locked"
    # instead of waiting - retry for up to 30s before giving up.
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    sql = (SCHEMA_DIR / "schema_tables.sql").read_text()
    conn.executescript(sql)


def create_indexes(conn: sqlite3.Connection) -> None:
    sql = (SCHEMA_DIR / "schema_indexes.sql").read_text()
    conn.executescript(sql)


def init_db(conn: sqlite3.Connection) -> None:
    create_tables(conn)
    create_indexes(conn)


def set_bulk_load_pragmas(conn: sqlite3.Connection) -> None:
    # This DB is a derived artifact, always rebuildable from the source CSVs
    # (never the source of truth) - so it's fine to trade crash-safety for
    # load speed here. synchronous=OFF skips fsync-per-commit entirely; if a
    # crash corrupts the file mid-load, just delete it and re-run ingestion.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -524288")  # ~512MB page cache
    conn.execute("PRAGMA mmap_size = 4000000000")  # 4GB, no-op if unsupported
