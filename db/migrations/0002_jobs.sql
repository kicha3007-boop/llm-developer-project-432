-- Таблица задач: деловое состояние, которое читает клиент.
-- История Temporal — внутренняя; «карточка на переделке в третий раз» живёт здесь.
CREATE TABLE jobs (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL DEFAULT 'card',
    idempotency_key TEXT UNIQUE,
    status          TEXT NOT NULL DEFAULT 'pending',
    payload         JSONB NOT NULL,
    result          JSONB,
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    review_reason   TEXT,
    workflow_id     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Поиск зависших задач: нетерминальный статус и давно не обновлялась
CREATE INDEX jobs_status_updated_idx ON jobs (status, updated_at);
