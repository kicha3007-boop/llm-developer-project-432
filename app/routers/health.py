"""Проверки состояния: живость — про процесс, готовность — про зависимости."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core import db

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Живость: процесс отвечает. Не зависит от базы — её остановка не повод перезапускать нас."""
    return {"status": "ok"}


@router.get("/ready")
def ready() -> JSONResponse:
    """Готовность: база отвечает и расширение vector активно."""
    try:
        state = db.readiness()
    except Exception as exc:
        return JSONResponse(
            {"status": "not ready", "db": f"error: {type(exc).__name__}"}, status_code=503
        )
    if not state.get("vector"):
        return JSONResponse({"status": "not ready", **state}, status_code=503)
    return JSONResponse({"status": "ready", **state})
