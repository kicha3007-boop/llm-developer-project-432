"""Разбор «грязного» ответа модели в JSON.

Модель оборачивает JSON в markdown-заборы, добавляет вступление и пояснение после.
``parse_llm_json`` это переживает и бросает ``LLMJsonError`` с понятной причиной, когда разобрать
нельзя: пустой ответ, JSON нет вовсе, JSON не закрыт.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMJsonError(ValueError):
    """Ответ модели не удалось разобрать как JSON."""


def _first_json_value(text: str) -> Any:
    """Первое корректное JSON-значение (объект или массив) в произвольном тексте."""
    decoder = json.JSONDecoder()
    opened = False
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        opened = True
        try:
            value, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        return value
    if opened:
        raise LLMJsonError("JSON в ответе модели не закрыт или повреждён")
    raise LLMJsonError("в ответе модели нет JSON")


def parse_llm_json(raw: str | None) -> Any:
    """JSON из ответа модели: чистый, в заборах или с текстом до и после."""
    if raw is None or not str(raw).strip():
        raise LLMJsonError("пустой ответ модели")
    text = _THINK_RE.sub("", str(raw)).strip()
    if not text:
        raise LLMJsonError("пустой ответ модели: только рассуждения, без JSON")
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)] + [text]
    last_error: LLMJsonError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return _first_json_value(candidate)
        except LLMJsonError as exc:
            last_error = exc
    raise last_error or LLMJsonError("в ответе модели нет JSON")


def parse_llm_object(raw: str | None) -> dict[str, Any]:
    value = parse_llm_json(raw)
    if not isinstance(value, dict):
        raise LLMJsonError(f"ожидался JSON-объект, получен {type(value).__name__}")
    return value
