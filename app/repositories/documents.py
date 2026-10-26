"""Репозиторий документов поставщиков. Соединение передаёт вызывающий."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg
from psycopg.rows import dict_row

_COLUMNS = (
    "d.id, d.content_hash, d.filename, d.kind, d.size_bytes, d.stored_path, d.status, d.error, "
    "d.title, d.pages, d.created_at, d.updated_at"
)


def create_document(
    conn: psycopg.Connection,
    *,
    document_id: str,
    filename: str,
    content_hash: str,
    kind: str,
    size_bytes: int | None = None,
    stored_path: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Регистрирует документ; то же содержимое повторно — существующая запись, без дубля."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO documents (id, content_hash, filename, kind, size_bytes, stored_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING id
            """,
            (document_id, content_hash, filename, kind.lstrip("."), size_bytes, stored_path),
        )
        created = cur.fetchone() is not None
        cur.execute(
            f"SELECT {_COLUMNS} FROM documents d WHERE d.content_hash = %s", (content_hash,)
        )
        row = cur.fetchone()
    assert row is not None
    return row, created


def get_document(conn: psycopg.Connection, document_id: str) -> dict[str, Any] | None:
    """Документ со счётчиками фрагментов (всего и с вектором)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT {_COLUMNS},
                   (SELECT count(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks_count,
                   (SELECT count(*) FROM chunks c
                     WHERE c.doc_id = d.id AND c.embedding IS NOT NULL) AS embedded_count
            FROM documents d WHERE d.id = %s
            """,
            (document_id,),
        )
        return cur.fetchone()


def get_documents(conn: psycopg.Connection, document_ids: Sequence[str]) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {_COLUMNS} FROM documents d WHERE d.id = ANY(%s) ORDER BY d.id",
            (list(document_ids),),
        )
        return cur.fetchall()


def set_status(
    conn: psycopg.Connection,
    document_id: str,
    status: str,
    *,
    error: str | None = None,
    title: str | None = None,
    pages: int | None = None,
) -> None:
    sets = ["status = %s", "updated_at = now()", "error = %s"]
    args: list[Any] = [status, error]
    if title is not None:
        sets.append("title = %s")
        args.append(title)
    if pages is not None:
        sets.append("pages = %s")
        args.append(pages)
    args.append(document_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE documents SET {', '.join(sets)} WHERE id = %s", args)


def mark_chunks_suspicious(conn: psycopg.Connection, chunk_ids: Sequence[str]) -> int:
    """Вердикт защиты сохраняется во фрагменте: заблокированный фрагмент отличим от обычного."""
    if not chunk_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute("UPDATE chunks SET suspicious = TRUE WHERE id = ANY(%s)", (list(chunk_ids),))
        return cur.rowcount
