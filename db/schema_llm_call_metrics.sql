CREATE TABLE IF NOT EXISTS llm_call_metrics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id         TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    call_type          TEXT NOT NULL,
    model              TEXT NOT NULL,
    prompt             TEXT NOT NULL,
    instructions       TEXT NOT NULL,
    answer             TEXT,
    prompt_tokens      INTEGER NOT NULL,
    completion_tokens  INTEGER NOT NULL,
    total_tokens       INTEGER NOT NULL,
    response_time_s    REAL NOT NULL,
    cost_usd           REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_created_at ON llm_call_metrics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_request_id ON llm_call_metrics(request_id);
