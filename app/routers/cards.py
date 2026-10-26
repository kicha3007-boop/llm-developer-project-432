"""Синхронная генерация карточки по тексту: весь пайплайн внутри запроса (для отладки)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.cards import CardRequest, CardResponse
from app.services.pipeline import run_pipeline
from app.services.structured import StructuredOutputError

router = APIRouter(tags=["cards"])


@router.post("/cards", response_model=CardResponse)
async def create_card(body: CardRequest) -> CardResponse:
    try:
        result = await run_pipeline(body.supplier_text)
    except StructuredOutputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return CardResponse(
        status=result.status,
        attempts=result.attempts,
        card=result.card,
        issues=result.issues,
        review_reason=result.review_reason,
    )
