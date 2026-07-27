import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent


def init_feedback_table(conn: sqlite3.Connection) -> None:
    sql = (SCHEMA_DIR / "schema_feedback.sql").read_text()
    conn.executescript(sql)


def insert_feedback(
    conn: sqlite3.Connection,
    request_id: str,
    score: int | None = None,
    source: str = "user",
    relevance: str | None = None,
    explanation: str | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO feedback (request_id, source, relevance, explanation, score, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            source,
            relevance,
            explanation,
            score,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def fetch_feedback(conn: sqlite3.Connection, request_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM feedback WHERE request_id = ? ORDER BY created_at", (request_id,)
    ).fetchall()
