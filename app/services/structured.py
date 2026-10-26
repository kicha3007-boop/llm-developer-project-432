"""Структурированный вывод: схема в запрос, разбор, валидация с лечением, точечная переделка.

Три уровня надёжности:
1. схема результата уходит в модель (``response_format``) и дублируется текстом в инструкции;
2. ответ разбирается терпимо к мусору (``json_utils``) и проверяется Pydantic-моделью;
3. невалидный ответ не выбрасывается: он уходит обратно в модель вместе с текстом ошибки.
   Если сломалось одно поле, переделывается только оно — с передачей текущего черновика.

Число попыток лечения ограничено (``llm_repair_attempts``), иначе лечение дороже генерации.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.agents import prompts
from app.agents.prompts import AgentSpec
from app.core.config import settings
from app.core.logging import get_logger
from app.llm import client as llm_client
from app.services.json_utils import JSONExtractionError, extract_json_object

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Модель не дала валидный ответ за отведённое число попыток лечения."""

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


def schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    return model_cls.model_json_schema()


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "карточка"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def single_broken_field(exc: ValidationError, model_cls: type[BaseModel]) -> str | None:
    """Имя поля, если все ошибки валидации относятся к одному полю верхнего уровня."""
    fields = {str(err["loc"][0]) if err["loc"] else "" for err in exc.errors()}
    if len(fields) == 1:
        (name,) = fields
        if name in model_cls.model_fields:
            return name
    return None


def parse_response[T: BaseModel](raw: str | None, model_cls: type[T]) -> T:
    """Разбор и валидация без лечения (для тестов и простых мест)."""
    return model_cls.model_validate(extract_json_object(raw))


async def repair_field(
    data: dict[str, Any], field: str, error: str, *, purpose: str
) -> dict[str, Any]:
    """Точечная переделка одного поля: модель видит текущий черновик и ошибку поля."""
    raw = await llm_client.run_agent(
        prompts.FIELD_REPAIR,
        prompts.build_field_repair_prompt(data, field, error),
        purpose=f"{purpose}:repair_field:{field}",
    )
    fixed = extract_json_object(raw)
    if field not in fixed:
        raise JSONExtractionError(f"в ответе нет поля «{field}»")
    return {**data, field: fixed[field]}


async def generate_structured[T: BaseModel](
    spec: AgentSpec,
    prompt: str,
    model_cls: type[T],
    *,
    purpose: str | None = None,
    max_repairs: int | None = None,
) -> T:
    """Генерирует ответ по схеме ``model_cls`` с лечением невалидных ответов."""
    purpose = purpose or spec.name
    repairs = settings.llm_repair_attempts if max_repairs is None else max_repairs
    schema = schema_for(model_cls)
    hint = prompts.schema_hint(schema)

    raw: str | None = await llm_client.run_agent(
        spec, prompt + hint, schema=schema, purpose=purpose
    )
    data: dict[str, Any] | None = None
    for attempt in range(repairs + 1):
        broken_field: str | None = None
        try:
            if raw is not None:
                data = extract_json_object(raw)
            return model_cls.model_validate(data)
        except JSONExtractionError as exc:
            error = str(exc)
        except ValidationError as exc:
            error = format_validation_error(exc)
            broken_field = single_broken_field(exc, model_cls)

        if attempt == repairs:
            log.warning("structured_failed", purpose=purpose, attempts=attempt + 1, error=error)
            raise StructuredOutputError(
                f"модель не дала валидный ответ за {attempt + 1} попыток: {error}", attempt + 1
            )

        if broken_field is not None and data is not None:
            log.info("structured_repair_field", purpose=purpose, field=broken_field, error=error)
            try:
                data = await repair_field(data, broken_field, error, purpose=purpose)
                raw = None  # следующая итерация проверяет исправленный черновик
                continue
            except JSONExtractionError as exc:
                error = f"{error}; переделка поля не удалась: {exc}"

        log.info("structured_repair", purpose=purpose, attempt=attempt + 1, error=error)
        raw = await llm_client.run_agent(
            spec,
            prompts.build_repair_prompt(
                raw if raw is not None else json.dumps(data, ensure_ascii=False), error
            )
            + hint,
            schema=schema,
            purpose=f"{purpose}:repair",
        )
    raise AssertionError("unreachable")  # pragma: no cover
