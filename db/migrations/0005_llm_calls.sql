-- Журнал вызовов модели: пишет клиент модели после каждого вызова.
-- Деньги — NUMERIC, а не float.
CREATE TABLE llm_calls (
    id            BIGSERIAL PRIMARY KEY,
    job_id        TEXT,
    request_id    TEXT,
    purpose       TEXT NOT NULL,
    tier          TEXT NOT NULL CHECK (tier IN ('main', 'cheap')),
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost          NUMERIC(14, 8) NOT NULL DEFAULT 0,
    duration_ms   INTEGER NOT NULL,
    attempt       INTEGER NOT NULL DEFAULT 1,
    status        TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX llm_calls_job_idx ON llm_calls (job_id);
CREATE INDEX llm_calls_model_idx ON llm_calls (model, created_at);
