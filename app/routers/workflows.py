"""Ручная проверка карточки: состояние процесса и сигналы подтверждения/отказа."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas.jobs import DecisionIn
from app.temporal import client as temporal

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.get("/{job_id}")
async def workflow_state(job_id: str) -> dict[str, Any]:
    try:
        return await temporal.describe(job_id)
    except temporal.WorkflowNotFound as exc:
        raise HTTPException(404, "процесс не найден") from exc


async def _signal(job_id: str, name: str, body: DecisionIn | None) -> dict[str, str]:
    try:
        await temporal.signal(job_id, name, body.comment if body else None)
    except temporal.WorkflowNotFound as exc:
        raise HTTPException(404, "процесс не найден") from exc
    return {"job_id": job_id, "signal": name}


@router.post("/{job_id}/approve", status_code=202)
async def approve(job_id: str, body: DecisionIn | None = None) -> dict[str, str]:
    return await _signal(job_id, "approve", body)


@router.post("/{job_id}/reject", status_code=202)
async def reject(job_id: str, body: DecisionIn | None = None) -> dict[str, str]:
    return await _signal(job_id, "reject", body)
