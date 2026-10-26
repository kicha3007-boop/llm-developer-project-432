-- Документы поставщиков и их фрагменты.
-- Идентификатор документа — префикс sha256 содержимого: повторная загрузка не создаёт дублей.
CREATE TABLE documents (
    id           TEXT PRIMARY KEY,
    sha256       TEXT NOT NULL UNIQUE,
    filename     TEXT NOT NULL,
    extension    TEXT NOT NULL,
    size_bytes   BIGINT NOT NULL,
    storage_path TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'new'
                 CHECK (status IN ('new', 'parsing', 'indexing', 'indexed', 'failed')),
    error        TEXT,
    title        TEXT,
    pages        INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    page        INTEGER,
    section     TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, ord)
);

CREATE INDEX chunks_document_idx ON chunks (document_id);
-- Фильтр по метаданным: артикул и бренд
CREATE INDEX chunks_articul_idx ON chunks ((metadata ->> 'articul'));
CREATE INDEX chunks_brand_idx ON chunks ((metadata ->> 'brand'));
