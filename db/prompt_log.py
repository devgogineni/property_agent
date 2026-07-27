import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent


def init_prompt_log_table(conn: sqlite3.Connection) -> None:
    sql = (SCHEMA_DIR / "schema_prompt_log.sql").read_text()
    conn.executescript(sql)


def insert_prompt_log(
    conn: sqlite3.Connection,
    question: str,
    answer: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO prompt_logs
            (created_at, question, answer, model, input_tokens, output_tokens, total_tokens, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            question,
            answer,
            model,
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            cost_usd,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def fetch_prompt_logs(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            p.*,
            COALESCE(SUM(CASE WHEN f.score > 0 THEN 1 ELSE 0 END), 0) AS feedback_positive,
            COALESCE(SUM(CASE WHEN f.score < 0 THEN 1 ELSE 0 END), 0) AS feedback_negative,
            MAX(CASE WHEN f.source = 'judge' THEN f.relevance END) AS judge_relevance,
            MAX(CASE WHEN f.source = 'judge' THEN f.explanation END) AS judge_explanation
        FROM prompt_logs p
        LEFT JOIN feedback f ON f.request_id = CAST(p.id AS TEXT)
        GROUP BY p.id
        ORDER BY p.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
