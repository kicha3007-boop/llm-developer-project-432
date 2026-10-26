"""Структурированные логи на structlog.

Поля из контекста (``request_id``, ``job_id``) привязываются один раз и попадают во все записи,
поэтому поиск по идентификатору задачи показывает весь её путь: обработчик, шаги процесса,
вызовы модели. Перед выводом каждая запись проходит через маскирование персональных данных —
вторая линия обороны, чтобы исходные контакты не попали в логи, даже если их кто-то залогирует.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import settings

_configured = False


def _mask_pii_processor(_logger: Any, _name: str, event_dict: dict) -> dict:
    # Импорт внутри: guardrails не нужен при импорте модуля логов
    from app.guardrails.pii import mask_value

    return {key: mask_value(value) for key, value in event_dict.items()}


def setup_logging(force: bool = False) -> None:
    """Настраивает логи процесса. Повторный вызов ничего не делает."""
    global _configured
    if _configured and not force:
        return
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if settings.log_file:
        handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)
    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _mask_pii_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)


def bind_context(**fields: Any) -> None:
    """Привязывает поля (request_id, job_id, ...) ко всем следующим записям в этом контексте."""
    structlog.contextvars.bind_contextvars(**{k: v for k, v in fields.items() if v is not None})


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


def current_context() -> dict[str, Any]:
    return structlog.contextvars.get_contextvars()
