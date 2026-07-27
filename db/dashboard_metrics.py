import sqlite3


def fetch_dashboard_summary(conn: sqlite3.Connection) -> dict:
    """Aggregate KPIs across all conversations, for the dashboard page.

    avg_response_time_s is per *conversation* (extraction call + answer call
    combined), not per individual LLM call - that's what a user actually
    waits for, and llm_call_metrics is the only table with latency.
    """
    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS total_conversations,
            COALESCE(AVG(total_tokens), 0) AS avg_tokens,
            COALESCE(SUM(cost_usd), 0) AS total_cost_usd
        FROM prompt_logs
        """
    ).fetchone()

    latency = conn.execute(
        """
        SELECT COALESCE(AVG(conversation_response_time_s), 0) AS avg_response_time_s
        FROM (
            SELECT SUM(response_time_s) AS conversation_response_time_s
            FROM llm_call_metrics
            GROUP BY request_id
        )
        """
    ).fetchone()

    feedback = conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END), 0) AS positive_feedback,
            COALESCE(SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END), 0) AS negative_feedback
        FROM feedback
        """
    ).fetchone()

    relevance = conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN relevance = 'RELEVANT' THEN 1 ELSE 0 END), 0) AS relevant_count,
            COALESCE(SUM(CASE WHEN relevance = 'PARTLY_RELEVANT' THEN 1 ELSE 0 END), 0) AS partly_relevant_count,
            COALESCE(SUM(CASE WHEN relevance = 'NON_RELEVANT' THEN 1 ELSE 0 END), 0) AS non_relevant_count
        FROM feedback
        WHERE source = 'judge'
        """
    ).fetchone()

    return {
        "total_conversations": totals["total_conversations"],
        "avg_tokens": totals["avg_tokens"],
        "total_cost_usd": totals["total_cost_usd"],
        "avg_response_time_s": latency["avg_response_time_s"],
        "positive_feedback": feedback["positive_feedback"],
        "negative_feedback": feedback["negative_feedback"],
        "relevant_count": relevance["relevant_count"],
        "partly_relevant_count": relevance["partly_relevant_count"],
        "non_relevant_count": relevance["non_relevant_count"],
    }


def fetch_conversation_series(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    """Per-conversation cost and response time for the most recent `limit`
    conversations, oldest first - for the dashboard's cost/response-time
    over time charts (mirrors the llm-zoomcamp streamlit dashboard's
    get_conversations(limit=100), adapted to this project's two-call-per-
    conversation shape by summing llm_call_metrics per request_id).
    """
    return conn.execute(
        """
        SELECT * FROM (
            SELECT
                p.id AS id,
                p.created_at AS created_at,
                p.cost_usd AS cost_usd,
                COALESCE(SUM(m.response_time_s), 0) AS response_time_s
            FROM prompt_logs p
            LEFT JOIN llm_call_metrics m ON m.request_id = CAST(p.id AS TEXT)
            GROUP BY p.id
            ORDER BY p.created_at DESC
            LIMIT ?
        ) recent
        ORDER BY created_at ASC
        """,
        (limit,),
    ).fetchall()
