"""Асинхронная генерация по тексту и чтение задач: статус, результат, стоимость."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.core.logging import bind_context, get_logger
from app.repositories import jobs as jobs_repo
from app.repositories import llm_calls as llm_calls_repo
from app.schemas.cards import CardRequest
from app.schemas.jobs import JobOut, JobStatus
from app.temporal import client as temporal

router = APIRouter(tags=["jobs"])
log = get_logger(__name__)

Starter = Callable[[str], Awaitable[str]]


async def start_job(
    kind: str, payload: dict[str, Any], idempotency_key: str | None, starter: Starter
) -> JSONResponse:
    """Сначала строка в базе, потом процесс: иначе воркер может не найти задачу.

    Повтор с тем же ключом идемпотентности возвращает первую задачу и процесс не запускает.
    """
    job_id, created = await run_in_threadpool(jobs_repo.create_job, kind, payload, idempotency_key)
    bind_context(job_id=job_id)  # дальше все записи обработчика несут идентификатор задачи
    log.info("job_accepted", kind=kind, created=created, idempotency_key=idempotency_key)
    if created:
        try:
            workflow_id = await starter(job_id)
        except Exception as exc:
            error = f"процесс не запущен: {type(exc).__name__}"
            await run_in_threadpool(
                lambda: jobs_repo.set_status(job_id, JobStatus.FAILED, error=error)
            )
            raise HTTPException(503, "движок процессов недоступен") from exc
        await run_in_threadpool(jobs_repo.set_workflow_id, job_id, workflow_id)
    job = await run_in_threadpool(jobs_repo.get_job, job_id)
    status = job["status"] if job else JobStatus.PENDING
    return JSONResponse({"job_id": job_id, "status": status}, status_code=202)


@router.post("/jobs", status_code=202)
async def create_job(
    body: CardRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    return await start_job(
        "card",
        {"supplier_text": body.supplier_text},
        idempotency_key,
        lambda job_id: temporal.start_card_workflow(job_id, body.supplier_text),
    )


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    row = jobs_repo.get_job(job_id)
    if row is None:
        raise HTTPException(404, "задача не найдена")
    return JobOut(
        job_id=row["id"],
        kind=row["kind"],
        status=row["status"],
        attempts=row["attempts"],
        result=row["result"],
        error=row["error"],
        review_reason=row["review_reason"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _plain(value: Any) -> Any:
    """Decimal — строкой без экспоненты («0.00000000», а не «0E-8»)."""
    return f"{value:f}" if isinstance(value, Decimal) else value


@router.get("/jobs/{job_id}/cost")
def get_job_cost(job_id: str) -> dict[str, Any]:
    """Стоимость одной карточки и разбивка вызовов по моделям."""
    if jobs_repo.get_job(job_id) is None:
        raise HTTPException(404, "задача не найдена")
    total = llm_calls_repo.cost_for_job(job_id)
    return {
        "job_id": job_id,
        "total": {k: _plain(v) for k, v in total.items()},
        "by_model": [
            {k: _plain(v) for k, v in row.items()}
            for row in llm_calls_repo.breakdown_by_model(job_id)
        ],
    }
