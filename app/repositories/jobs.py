"""Репозиторий задач — единственный владелец таблицы ``jobs``. SQL живёт только здесь."""

from __future__ import annotations

import uuid
from typing import Any

from psycopg.types.json import Jsonb

from app.core.db import connection


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def create_job(
    kind: str, payload: dict[str, Any], idempotency_key: str | None = None
) -> tuple[str, bool]:
    """Создаёт задачу в статусе pending. Возвращает (id, создана_ли_новая).

    Повторный запрос с тем же ключом идемпотентности возвращает существующую задачу.
    Гонка двух одновременных запросов закрыта ограничением уникальности и ON CONFLICT.
    """
    job_id = new_job_id()
    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO jobs (id, kind, idempotency_key, payload)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (job_id, kind, idempotency_key, Jsonb(payload)),
        ).fetchone()
        if row:
            return row["id"], True
        existing = conn.execute(
            "SELECT id FROM jobs WHERE idempotency_key = %s", (idempotency_key,)
        ).fetchone()
    if existing is None:  # pragma: no cover - ключ удалили между запросами
        raise RuntimeError("Задача с ключом идемпотентности исчезла")
    return existing["id"], False


def get_job(job_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute(
            """
            SELECT id, kind, idempotency_key, status, payload, result, attempts, error,
                   review_reason, workflow_id, created_at, updated_at
            FROM jobs WHERE id = %s
            """,
            (job_id,),
        ).fetchone()


def set_status(
    job_id: str,
    status: str,
    *,
    result: dict[str, Any] | None = None,
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
        args.append(Jsonb(result))
    if error is not None:
        sets.append("error = %s")
        args.append(error)
    if review_reason is not None:
        sets.append("review_reason = %s")
        args.append(review_reason)
    if bump_attempts:
        sets.append("attempts = attempts + 1")
    args.append(job_id)
    with connection() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s", args)


def set_workflow_id(job_id: str, workflow_id: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE jobs SET workflow_id = %s, updated_at = now() WHERE id = %s",
            (workflow_id, job_id),
        )


def find_stuck(older_than_minutes: int = 15) -> list[dict[str, Any]]:
    """Задачи в нетерминальном статусе, которые давно никто не обновлял."""
    with connection() as conn:
        return conn.execute(
            """
            SELECT id, status, attempts, updated_at FROM jobs
            WHERE status NOT IN ('confirmed', 'rejected', 'failed', 'approved', 'needs_review')
              AND updated_at < now() - make_interval(mins => %s)
            ORDER BY updated_at
            """,
            (older_than_minutes,),
        ).fetchall()
