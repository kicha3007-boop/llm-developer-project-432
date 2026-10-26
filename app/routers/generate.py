"""Главный обработчик: карточка товара по загруженным документам."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.repositories import documents as documents_repo
from app.routers.jobs import start_job
from app.schemas.cards import GenerateCardRequest
from app.temporal import client as temporal

router = APIRouter(tags=["generate"])


@router.post("/generate-card", status_code=202)
async def generate_card(
    body: GenerateCardRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    ids = list(dict.fromkeys(body.document_ids))
    found = {d["id"] for d in await run_in_threadpool(documents_repo.get_many, ids)}
    unknown = [i for i in ids if i not in found]
    if unknown:
        raise HTTPException(404, f"документы не найдены: {', '.join(unknown)}")
    return await start_job(
        "rag",
        {"document_ids": ids, "product_hint": body.product_hint},
        idempotency_key,
        lambda job_id: temporal.start_generate_workflow(job_id, ids, body.product_hint),
    )
