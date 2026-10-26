-- Имена колонок по контракту эталона курса: documents(content_hash, kind, stored_path),
-- chunks(doc_id, ordinal). Служебные колонки, которых нет в минимальной вставке, получают
-- значения по умолчанию или допускают NULL. Флаг suspicious хранит вердикт защиты.
ALTER TABLE documents RENAME COLUMN sha256 TO content_hash;
ALTER TABLE documents RENAME COLUMN extension TO kind;
ALTER TABLE documents RENAME COLUMN storage_path TO stored_path;
ALTER TABLE documents ALTER COLUMN stored_path DROP NOT NULL;
ALTER TABLE documents ALTER COLUMN size_bytes DROP NOT NULL;
UPDATE documents SET kind = ltrim(kind, '.');

ALTER TABLE chunks RENAME COLUMN document_id TO doc_id;
ALTER TABLE chunks RENAME COLUMN ord TO ordinal;
ALTER TABLE chunks ALTER COLUMN kind SET DEFAULT 'text';
ALTER TABLE chunks ADD COLUMN suspicious BOOLEAN NOT NULL DEFAULT FALSE;

ALTER INDEX chunks_document_idx RENAME TO chunks_doc_idx;
