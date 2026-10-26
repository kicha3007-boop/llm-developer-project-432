"""Репозиторий документов поставщиков."""

from __future__ import annotations

from typing import Any

from app.core.db import connection

_COLUMNS = (
    "id, sha256, filename, extension, size_bytes, storage_path, status, error, title, pages, "
    "created_at, updated_at"
)


def get_or_create(
    *,
    document_id: str,
    sha256: str,
    filename: str,
    extension: str,
    size_bytes: int,
    storage_path: str,
) -> tuple[dict[str, Any], bool]:
    """Регистрирует документ; повторная загрузка того же содержимого возвращает существующий."""
    with connection() as conn:
        row = conn.execute(
            f"""
            INSERT INTO documents (id, sha256, filename, extension, size_bytes, storage_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (sha256) DO NOTHING
            RETURNING {_COLUMNS}
            """,
            (document_id, sha256, filename, extension, size_bytes, storage_path),
        ).fetchone()
        if row:
            return row, True
        existing = conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE sha256 = %s", (sha256,)
        ).fetchone()
    assert existing is not None
    return existing, False


def get(document_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE id = %s", (document_id,)
        ).fetchone()


def get_many(document_ids: list[str]) -> list[dict[str, Any]]:
    with connection() as conn:
        return conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE id = ANY(%s) ORDER BY id",
            (document_ids,),
        ).fetchall()


def set_status(
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
    with connection() as conn:
        conn.execute(f"UPDATE documents SET {', '.join(sets)} WHERE id = %s", args)
