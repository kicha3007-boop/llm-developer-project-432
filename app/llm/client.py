"""Клиент модели — единственное место, которое знает про провайдера.

Политика доступа к модели:
- таймаут на HTTP-запрос и на весь прогон агента;
- повтор с растущей задержкой и случайной примесью — только на ошибках, которые повтор лечит
  (сеть, таймаут, 429, 5xx); ошибка в запросе (400, 401, 404) не повторяется;
- выбор модели: служебные роли (``AgentSpec.cheap``) идут на дешёвую модель;
- учёт: после каждого вызова в таблицу ``llm_calls`` пишутся задача, модель, токены, стоимость
  и длительность. Учёт живёт здесь, а не в вызывающем коде, — его нельзя забыть в новом месте.

Здесь повторяется ВЫЗОВ модели. Повтор шага целиком — работа движка процессов (Temporal).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.agents.prompts import AgentSpec
from app.core import db
from app.core.config import settings
from app.core.logging import current_context, get_logger
from app.llm.cost import cost_for_call

log = get_logger(__name__)

_openai_client: AsyncOpenAI | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


def _client() -> AsyncOpenAI:
    """Клиент SDK создаётся один раз на процесс. Собственные повторы SDK выключены:
    уровень повторов вызова один — цикл в ``run_agent``."""
    global _openai_client
    if _openai_client is None:
        from agents import set_tracing_disabled

        _openai_client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "not-needed",
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        # Трассировки SDK уходят на адрес OpenAI мимо base_url — отключаем
        set_tracing_disabled(True)
    return _openai_client


def model_for(spec: AgentSpec) -> tuple[str, str]:
    """Уровень и имя модели для роли: служебные вызовы — на дешёвую модель."""
    if spec.cheap:
        return "cheap", settings.llm_cheap_model
    return "main", settings.llm_model


def _retryable(exc: BaseException) -> bool:
    """Повторяем то, что может измениться само: сеть, таймаут, перегрузка, сбой сервера."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def _backoff(attempt: int) -> float:
    """Экспонента с потолком и случайной примесью, чтобы воркеры не просыпались синхронно."""
    return min(2**attempt * 0.5, 8.0) * (1 + random.random() * 0.25)


async def _call_model(
    spec: AgentSpec, model: str, prompt: str, schema: dict[str, Any] | None
) -> LLMResponse:
    """Один прогон агента OpenAI Agents SDK через chat completions совместимого сервера."""
    from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner

    extra_body: dict[str, Any] | None = None
    if schema is not None and settings.llm_structured_output == "json_schema":
        extra_body = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": spec.name, "schema": schema, "strict": False},
            }
        }
    agent = Agent(
        name=spec.name,
        instructions=spec.instructions,
        model=OpenAIChatCompletionsModel(model=model, openai_client=_client()),
        model_settings=ModelSettings(
            temperature=spec.temperature
            if spec.temperature is not None
            else settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            extra_body=extra_body,
        ),
    )
    result = await Runner.run(agent, prompt, max_turns=1)
    usage = result.context_wrapper.usage
    output = result.final_output
    return LLMResponse(
        text=output if isinstance(output, str) else str(output or ""),
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
        model=model,
    )


async def _record_call(**fields: Any) -> None:
    """Пишет вызов в таблицу llm_calls. Сбой учёта не роняет генерацию, но виден в логе.

    Запись идёт, только если процесс уже поднял пул соединений (приложение или воркер);
    модульные тесты без базы сюда не ходят.
    """
    if not db.pool_initialized():
        return
    from app.repositories import llm_calls

    try:
        await asyncio.to_thread(llm_calls.insert_call, **fields)
    except Exception as exc:  # учёт не должен ломать основной путь
        log.warning("llm_call_record_failed", error=repr(exc))


async def run_agent(
    spec: AgentSpec,
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    purpose: str | None = None,
) -> str:
    """Вызывает модель для роли и возвращает текст ответа.

    ``schema`` — JSON-схема ожидаемого ответа; при поддержке сервером передаётся в
    ``response_format``. Разбор и проверку ответа делает ``app/services/structured.py``.
    """
    tier, model = model_for(spec)
    ctx = current_context()
    job_id, request_id = ctx.get("job_id"), ctx.get("request_id")
    purpose = purpose or spec.name
    max_attempts = settings.llm_max_retries
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                _call_model(spec, model, prompt, schema), timeout=settings.llm_timeout_s
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            await _record_call(
                job_id=job_id,
                request_id=request_id,
                purpose=purpose,
                tier=tier,
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost=cost_for_call(tier, 0, 0),
                duration_ms=duration_ms,
                attempt=attempt,
                status="error",
                error=repr(exc)[:500],
            )
            if attempt == max_attempts or not _retryable(exc):
                log.error(
                    "llm_call_failed",
                    agent=spec.name,
                    model=model,
                    attempt=attempt,
                    error=repr(exc),
                )
                raise
            delay = _backoff(attempt)
            log.warning(
                "llm_call_retry",
                agent=spec.name,
                model=model,
                attempt=attempt,
                delay_s=round(delay, 2),
                error=repr(exc),
            )
            await asyncio.sleep(delay)
            continue

        duration_ms = int((time.perf_counter() - started) * 1000)
        cost = cost_for_call(tier, response.input_tokens, response.output_tokens)
        await _record_call(
            job_id=job_id,
            request_id=request_id,
            purpose=purpose,
            tier=tier,
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=cost,
            duration_ms=duration_ms,
            attempt=attempt,
            status="ok",
            error=None,
        )
        log.info(
            "llm_call",
            agent=spec.name,
            purpose=purpose,
            tier=tier,
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=str(cost),
            duration_ms=duration_ms,
            attempt=attempt,
        )
        return response.text
    raise RuntimeError("unreachable")  # pragma: no cover
