"""Репозиторий журнала вызовов модели. Соединение передаёт вызывающий."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row


def insert_call(
    conn: psycopg.Connection,
    *,
    job_id: str | None,
    request_id: str | None,
    purpose: str,
    tier: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: Decimal,
    duration_ms: int,
    attempt: int,
    status: str,
    error: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_calls (job_id, request_id, purpose, tier, model, input_tokens,
                                   output_tokens, cost, duration_ms, attempt, status, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                job_id,
                request_id,
                purpose,
                tier,
                model,
                input_tokens,
                output_tokens,
                cost,
                duration_ms,
                attempt,
                status,
                error,
            ),
        )


def cost_for_job(conn: psycopg.Connection, job_id: str) -> dict[str, Any]:
    """Стоимость одной карточки — одним запросом по идентификатору задачи."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT count(*) AS calls,
                   coalesce(sum(cost), 0) AS cost,
                   coalesce(sum(input_tokens), 0) AS input_tokens,
                   coalesce(sum(output_tokens), 0) AS output_tokens,
                   coalesce(sum(cost) FILTER (WHERE tier = 'main'), 0) AS main_cost,
                   coalesce(sum(cost) FILTER (WHERE tier = 'cheap'), 0) AS cheap_cost
            FROM llm_calls WHERE job_id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else {}


def breakdown_by_model(conn: psycopg.Connection, job_id: str | None = None) -> list[dict[str, Any]]:
    """Разбивка по моделям: сколько вызовов и на какую сумму ушло на дешёвую и основную."""
    where, args = ("WHERE job_id = %s", (job_id,)) if job_id else ("", ())
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT tier, model, count(*) AS calls, sum(cost) AS cost,
                   sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
                   round(avg(duration_ms)) AS avg_ms
            FROM llm_calls {where}
            GROUP BY tier, model ORDER BY tier, model
            """,
            args,
        )
        return cur.fetchall()
