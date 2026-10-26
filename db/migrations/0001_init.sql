-- Активируем уже собранное в образе расширение pgvector и заводим таблицу,
-- в которой раннер миграций отмечает применённые версии.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
