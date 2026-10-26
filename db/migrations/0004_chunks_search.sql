-- Поиск по фрагментам: векторная колонка (размерность модели эмбеддингов из настроек — 768)
-- и колонка для полнотекстового поиска. Метрика — косинусное расстояние, индекс строится под неё.
ALTER TABLE chunks ADD COLUMN embedding vector(768);
ALTER TABLE chunks ADD COLUMN embedding_model TEXT;

-- 'simple' вместо 'russian': стеммер не режет артикулы и единицы, а русская морфология
-- закрывается векторной веткой гибридного поиска (см. docs/adr/0003-retrieval.md).
ALTER TABLE chunks ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(section, '') || ' ' || text)) STORED;

CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin_idx ON chunks USING gin (tsv);
-- Досчёт векторов ищет фрагменты без эмбеддинга
CREATE INDEX chunks_missing_embedding_idx ON chunks (id) WHERE embedding IS NULL;
