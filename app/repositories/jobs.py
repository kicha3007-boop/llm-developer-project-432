"""Репозиторий задач — единственный владелец таблицы ``jobs``. SQL живёт только здесь.

Сигнатуры — как в эталоне курса (``create_job(conn, payload, idempotency_key)``,
``set_status(conn, job_id, status, *, result, error, bump_attempts)``, ``get_job(conn, job_id)``):
соединение передаёт вызывающий — оно берётся из пула на одну операцию.
"""

from __future__ import annotations

import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel

_COLUMNS = (
    "id, kind, idempotency_key, status, payload, result, attempts, error, review_reason, "
    "workflow_id, created_at, updated_at"
)


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def create_job(
    conn: psycopg.Connection,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
    kind: str = "card",
) -> str:
    """Создаёт задачу в статусе pending и возвращает её id.

    Повторный запрос с тем же ключом идемпотентности возвращает существующую задачу: гонку
    двух одновременных запросов закрывают ограничение уникальности и ON CONFLICT.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO jobs (id, kind, idempotency_key, payload)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (new_job_id(), kind, idempotency_key, Jsonb(payload)),
        )
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute("SELECT id FROM jobs WHERE idempotency_key = %s", (idempotency_key,))
        existing = cur.fetchone()
    if existing is None:  # pragma: no cover - ключ удалили между запросами
        raise RuntimeError("задача с ключом идемпотентности исчезла")
    return existing["id"]


def get_job(conn: psycopg.Connection, job_id: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def get_job_by_key(conn: psycopg.Connection, idempotency_key: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {_COLUMNS} FROM jobs WHERE idempotency_key = %s", (idempotency_key,))
        return cur.fetchone()


def set_status(
    conn: psycopg.Connection,
    job_id: str,
    status: str,
    *,
    result: BaseModel | dict[str, Any] | None = None,
    error: str | None = None,
    review_reason: str | None = None,
    bump_attempts: bool = False,
) -> None:
    """Обновляет статус и время изменения; остальные поля — только если переданы.

    Имена колонок зашиты в коде, значения идут параметрами — сборка строки безопасна.
    Счётчик попыток растёт на стороне базы, а не чтением с последующей записью.
    """
    sets = ["status = %s", "updated_at = now()"]
    args: list[Any] = [status]
    if result is not None:
        sets.append("result = %s")
        args.append(Jsonb(result.model_dump() if isinstance(result, BaseModel) else result))
    if error is not None:
        sets.append("error = %s")
        args.append(error)
    if review_reason is not None:
        sets.append("review_reason = %s")
        args.append(review_reason)
    if bump_attempts:
        sets.append("attempts = attempts + 1")
    args.append(job_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s", args)


def set_workflow_id(conn: psycopg.Connection, job_id: str, workflow_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET workflow_id = %s, updated_at = now() WHERE id = %s",
            (workflow_id, job_id),
        )


def find_stuck(conn: psycopg.Connection, older_than_minutes: int = 15) -> list[dict[str, Any]]:
    """Задачи в незавершённом статусе, которые давно никто не обновлял."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, status, attempts, updated_at FROM jobs
            WHERE status NOT IN ('confirmed', 'rejected', 'failed', 'done', 'needs_review')
              AND updated_at < now() - make_interval(mins => %s)
            ORDER BY updated_at
            """,
            (older_than_minutes,),
        )
        return cur.fetchall()
