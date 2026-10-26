"""Сборка FastAPI-приложения: жизненный цикл, сквозной идентификатор запроса, роутеры."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.core.db import close_pool, open_pool
from app.core.logging import bind_context, clear_context, get_logger, setup_logging
from app.routers import cards, documents, generate, health, jobs, workflows

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Пул — ресурс уровня приложения: создаётся на старте, закрывается при остановке.
    # Первого соединения не ждём: живость не должна зависеть от базы.
    setup_logging()
    open_pool(wait=False)
    log.info("app_started")
    yield
    close_pool()
    log.info("app_stopped")


app = FastAPI(title="product-card-generator", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Идентификатор запроса рождается на входе и попадает во все записи лога по нему."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    clear_context()
    bind_context(request_id=request_id)
    log.info("http_request", method=request.method, path=request.url.path)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log.info("http_response", status=response.status_code, path=request.url.path)
    return response


app.include_router(health.router)
app.include_router(cards.router)
app.include_router(jobs.router)
app.include_router(documents.router)
app.include_router(generate.router)
app.include_router(workflows.router)
