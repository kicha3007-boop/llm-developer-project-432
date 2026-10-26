"""Структурированный вывод: валидация с лечением, точечная переделка поля, порог уверенности.

- ``validate_or_retry(prompt)`` — ответ модели разбирается терпимо к мусору и проверяется
  ``CardDraft``. Невалидный ответ не выбрасывается: он уходит обратно в модель вместе с текстом
  ошибки. Если сломалось одно поле, переделывается только оно (``regenerate_field``).
  Число исправлений ограничено (``LLM_REPAIR_ATTEMPTS``), дальше — ``CardValidationError``.
- ``regenerate_field(card, field, reason)`` — модель получает текущий черновик и просьбу
  исправить одно поле; остальные поля не трогаются.
- ``confidence_gate(card)`` — ниже порога уверенности задача ждёт человека (``needs_review``).

К модели — только через ``run_agent`` клиента (подменяется в тестах на уровне модуля).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.agents import prompts
from app.agents.prompts import field_fixer_agent, generator_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import run_agent
from app.schemas.cards import CardDraft
from app.schemas.jobs import JobStatus
from app.services.json_utils import LLMJsonError, parse_llm_object

log = get_logger(__name__)

M = TypeVar("M", bound=BaseModel)
Runner = Callable[[Any, str], Awaitable[Any]]


class CardValidationError(Exception):
    """Модель не дала валидный ответ за отведённое число исправлений."""

    def __init__(self, message: str, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "карточка"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _raw_text(output: Any) -> str:
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)


def _as_dict(output: Any) -> dict[str, Any]:
    if isinstance(output, BaseModel):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    return parse_llm_object(output)


def to_model[M: BaseModel](output: Any, model_cls: type[M]) -> M:
    """Ответ SDK бывает готовым объектом или строкой — приводим к одному типу."""
    if isinstance(output, model_cls):
        return output
    return model_cls.model_validate(_as_dict(output))


def _single_broken_field(exc: ValidationError, model_cls: type[BaseModel]) -> str | None:
    fields = {str(err["loc"][0]) if err["loc"] else "" for err in exc.errors()}
    if len(fields) == 1:
        (name,) = fields
        if name in model_cls.model_fields:
            return name
    return None


async def ask_model[M: BaseModel](
    agent: Any,
    prompt: str,
    model_cls: type[M],
    *,
    runner: Runner | None = None,
    max_retries: int | None = None,
) -> M:
    """Ответ модели по схеме ``model_cls`` с лечением: ошибка уходит обратно в модель."""
    call = runner or run_agent
    repairs = settings.llm_repair_attempts if max_retries is None else max_retries
    output = await call(agent, prompt)
    data: dict[str, Any] | None = None
    for attempt in range(repairs + 1):
        broken: str | None = None
        try:
            if output is not None:
                data = _as_dict(output)
            return model_cls.model_validate(data)
        except LLMJsonError as exc:
            error = str(exc)
        except ValidationError as exc:
            error = format_validation_error(exc)
            broken = _single_broken_field(exc, model_cls)
        if attempt == repairs:
            log.warning("structured_failed", agent=getattr(agent, "name", ""), error=error)
            raise CardValidationError(
                f"модель не дала валидный ответ за {attempt + 1} попыток: {error}", attempt + 1
            )
        if broken is not None and data is not None and model_cls is CardDraft:
            log.info("structured_repair_field", field=broken, error=error)
            fixed = await call(field_fixer_agent, prompts.build_field_prompt(data, broken, error))
            try:
                data = {**data, broken: _as_dict(fixed)[broken]}
                output = None  # следующая итерация проверяет исправленный черновик
                continue
            except (LLMJsonError, KeyError):
                pass
        log.info("structured_repair", agent=getattr(agent, "name", ""), error=error)
        output = await call(
            agent, prompts.build_repair_prompt(prompt, _raw_text(output or data), error)
        )
    raise AssertionError("unreachable")  # pragma: no cover


async def validate_or_retry(
    prompt: str,
    *,
    agent: Any = None,
    runner: Runner | None = None,
    max_retries: int | None = None,
) -> CardDraft:
    """Генерация черновика по схеме ``CardDraft`` с лечением невалидного ответа."""
    return await ask_model(
        agent or generator_agent, prompt, CardDraft, runner=runner, max_retries=max_retries
    )


async def regenerate_field(
    card: CardDraft, field: str, reason: str, *, runner: Runner | None = None
) -> CardDraft:
    """Точечная переделка: исправляется одно поле, остальные берутся из текущего черновика."""
    call = runner or run_agent
    output = await call(field_fixer_agent, prompts.build_field_prompt(card, field, reason))
    try:
        value = _as_dict(output)[field]
    except (LLMJsonError, KeyError) as exc:
        raise CardValidationError(f"в ответе нет поля «{field}»") from exc
    return CardDraft.model_validate({**card.model_dump(), field: value})


def confidence_gate(card: CardDraft) -> str:
    """Ниже порога уверенности карточку ждёт человек, а не «готово»."""
    if card.confidence < settings.confidence_threshold:
        return JobStatus.NEEDS_REVIEW
    return JobStatus.DONE
