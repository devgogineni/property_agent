import sqlite3
from pathlib import Path

from utils.evaluation_utils import LLMCallRecord

SCHEMA_DIR = Path(__file__).resolve().parent


def init_llm_call_metrics_table(conn: sqlite3.Connection) -> None:
    sql = (SCHEMA_DIR / "schema_llm_call_metrics.sql").read_text()
    conn.executescript(sql)


def insert_llm_call_metric(conn: sqlite3.Connection, request_id: str, call: LLMCallRecord) -> None:
    conn.execute(
        """
        INSERT INTO llm_call_metrics
            (request_id, created_at, call_type, model, prompt, instructions, answer,
             prompt_tokens, completion_tokens, total_tokens, response_time_s, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            call.timestamp.isoformat(),
            call.call_type,
            call.model,
            call.prompt,
            call.instructions,
            call.answer,
            call.prompt_tokens,
            call.completion_tokens,
            call.total_tokens,
            call.response_time,
            call.cost,
        ),
    )
    conn.commit()


def fetch_llm_call_metrics(
    conn: sqlite3.Connection, request_id: str | None = None, limit: int = 200
) -> list[sqlite3.Row]:
    if request_id is not None:
        return conn.execute(
            "SELECT * FROM llm_call_metrics WHERE request_id = ? ORDER BY created_at",
            (request_id,),
        ).fetchall()

    return conn.execute(
        "SELECT * FROM llm_call_metrics ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
