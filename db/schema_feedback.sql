CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL,
    source        TEXT NOT NULL,
    relevance     TEXT,
    explanation   TEXT,
    score         INTEGER,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_request_id ON feedback(request_id);
