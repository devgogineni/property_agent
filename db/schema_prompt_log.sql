CREATE TABLE IF NOT EXISTS prompt_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    question        TEXT NOT NULL,
    answer          TEXT,
    model           TEXT NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    total_tokens    INTEGER NOT NULL,
    cost_usd        REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_logs_created_at ON prompt_logs(created_at DESC);
