"""Клиент модели — единственное место, которое знает про провайдера.

Как в эталоне курса (урок «Разбор примера», ``setup_llm`` / ``_retryable`` / ``run_agent``):
- ``setup_llm()`` один раз на процесс настраивает OpenAI Agents SDK: клиент с адресом, ключом и
  таймаутом, интерфейс chat completions (его понимают локальные серверы), без трассировок;
- ``run_agent(agent, prompt)`` — таймаут на весь прогон и повтор с растущей задержкой и
  случайной примесью только на ошибках, которые повтор лечит (сеть, таймаут, 429, 5xx);
- учёт: после каждого вызова в ``llm_calls`` пишутся задача, модель, токены, стоимость и
  длительность (``_log_call`` / ``_log_failure``) — учёт нельзя забыть в новом месте.

Здесь повторяется ВЫЗОВ модели. Повтор шага целиком — работа движка процессов (Temporal).
"""

from __future__ import annotations

import asyncio
import random
import time
from contextvars import ContextVar
from typing import Any

from agents import Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.core import db
from app.core.config import settings
from app.core.logging import current_context, get_logger
from app.llm.cost import cost_for_call

log = get_logger(__name__)

_client_configured = False
# номер текущей попытки — для записи успешного вызова в llm_calls
_attempt: ContextVar[int] = ContextVar("llm_attempt", default=1)


def setup_llm() -> None:
    """Настройка SDK; безопасна к повторному вызову (API, воркер, команды)."""
    global _client_configured
    if _client_configured:
        return
    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "not-needed",
        timeout=settings.llm_timeout_s,
        max_retries=0,  # уровень повторов вызова один — цикл в run_agent
    )
    set_default_openai_client(client)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    _client_configured = True


def _retryable(exc: BaseException) -> bool:
    """Повторяем то, что может измениться само: сеть, таймаут, перегрузка, сбой сервера."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        return status == 429 or status >= 500
    return False


def _backoff(attempt: int) -> float:
    return min(2**attempt * 0.5, 8.0) * (1 + random.random() * 0.25)


def model_tier(agent: Any) -> tuple[str, str]:
    """Уровень и имя модели агента: служебные агенты созданы на дешёвой модели."""
    model = getattr(agent, "model", None)
    name = model if isinstance(model, str) else getattr(model, "model", None) or settings.llm_model
    tier = "cheap" if name == settings.llm_cheap_model and name != settings.llm_model else "main"
    return tier, name


def _usage(result: Any) -> tuple[int, int]:
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def _record(fields: dict[str, Any]) -> None:
    """Пишет вызов в llm_calls, если процесс поднял пул (API или воркер). Сбой учёта не
    роняет генерацию, но виден в логе."""
    if not db.pool_initialized():
        return
    from app.repositories import llm_calls

    try:
        with db.connection() as conn:
            llm_calls.insert_call(conn, **fields)
    except Exception as exc:  # учёт не должен ломать основной путь
        log.warning("llm_call_record_failed", error=repr(exc))


def _base_fields(agent: Any, started: float, attempt: int) -> dict[str, Any]:
    ctx = current_context()
    tier, model = model_tier(agent)
    return {
        "job_id": ctx.get("job_id"),
        "request_id": ctx.get("request_id"),
        "purpose": getattr(agent, "name", "agent"),
        "tier": tier,
        "model": model,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "attempt": attempt,
    }


def _log_call(agent: Any, result: Any, started: float) -> None:
    attempt = _attempt.get()
    fields = _base_fields(agent, started, attempt)
    input_tokens, output_tokens = _usage(result)
    cost = cost_for_call(fields["tier"], input_tokens, output_tokens)
    fields.update(
        input_tokens=input_tokens, output_tokens=output_tokens, cost=cost, status="ok", error=None
    )
    log.info(
        "llm_call",
        agent=fields["purpose"],
        tier=fields["tier"],
        model=fields["model"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=str(cost),
        duration_ms=fields["duration_ms"],
        attempt=attempt,
    )
    _record(fields)


def _log_failure(agent: Any, exc: BaseException, started: float, attempt: int) -> None:
    fields = _base_fields(agent, started, attempt)
    fields.update(
        input_tokens=0,
        output_tokens=0,
        cost=cost_for_call(fields["tier"], 0, 0),
        status="error",
        error=repr(exc)[:500],
    )
    _record(fields)


async def run_agent(agent: Any, prompt: str) -> Any:
    """Прогон агента с таймаутом и повторами. Возвращает ``final_output`` (обычно строку)."""
    setup_llm()
    max_retries = settings.llm_max_retries
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        _attempt.set(attempt)
        try:
            result = await asyncio.wait_for(
                Runner.run(agent, prompt), timeout=settings.llm_timeout_s
            )
        except Exception as exc:
            _log_failure(agent, exc, started, attempt)
            if attempt == max_retries or not _retryable(exc):
                log.error(
                    "llm_call_failed",
                    agent=getattr(agent, "name", "agent"),
                    attempt=attempt,
                    error=repr(exc),
                )
                raise
            delay = _backoff(attempt)
            log.warning(
                "llm_call_retry",
                agent=getattr(agent, "name", "agent"),
                attempt=attempt,
                delay_s=round(delay, 2),
                error=repr(exc),
            )
            await asyncio.sleep(delay)
            continue
        _log_call(agent, result, started)
        return result.final_output
    raise RuntimeError("unreachable")  # pragma: no cover
