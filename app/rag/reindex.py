"""Досчёт векторов фрагментам, у которых их нет.

Пригодится после сбоя и после смены модели (сбросить ``embedding`` и запустить заново).
Повторный запуск ничего не пересчитывает: берутся только фрагменты с ``embedding IS NULL``.

Команда: ``python -m app.rag.reindex`` (или ``make reindex``).
"""

from __future__ import annotations

import sys

from app.core.config import settings
from app.core.db import close_pool, connection, open_pool
from app.core.logging import get_logger, setup_logging
from app.rag.embedder import document_input, get_embedder
from app.repositories import chunks as chunks_repo

log = get_logger(__name__)


def reindex_missing(document_id: str | None = None, batch_size: int | None = None) -> int:
    """Считает векторы пачками и пишет их пачкой. Возвращает число проиндексированных.

    Соединение берётся на каждую короткую операцию с базой, а не на всё время счёта векторов.
    """
    batch_size = batch_size or settings.embedding_batch_size
    embedder = get_embedder()
    done = 0
    while True:
        with connection() as conn:
            rows = chunks_repo.missing_embeddings(conn, batch_size, doc_id=document_id)
        if not rows:
            break
        texts = [document_input(r["text"], r["section"], r["title"]) for r in rows]
        vectors = embedder.embed_documents(texts)
        with connection() as conn:
            chunks_repo.save_embeddings(
                conn,
                [(r["id"], v) for r, v in zip(rows, vectors, strict=True)],
                embedder.model_name,
            )
        done += len(rows)
        log.info("reindex_progress", indexed=done, batch=len(rows), document_id=document_id)
    log.info("reindex_done", indexed=done, document_id=document_id)
    return done


def main() -> int:
    setup_logging()
    open_pool(wait=True)
    try:
        count = reindex_missing()
    finally:
        close_pool()
    print(f"Проиндексировано фрагментов: {count}" if count else "Новых фрагментов без векторов нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
