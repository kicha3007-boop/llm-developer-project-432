"""Синхронная генерация карточки по тексту: весь пайплайн внутри запроса (для отладки)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.cards import CardResponse, JobIn
from app.services.pipeline import run_pipeline
from app.services.structured import CardValidationError

router = APIRouter(tags=["cards"])


@router.post("/cards", response_model=CardResponse)
async def create_card(body: JobIn) -> CardResponse:
    try:
        draft, attempts, verdict, status = await run_pipeline(body.supplier_text)
    except CardValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return CardResponse(status=status, attempts=attempts, verdict=verdict, card=draft)
