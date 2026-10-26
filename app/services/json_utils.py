"""Разбор грязного ответа модели в JSON.

Модель оборачивает JSON в markdown-заборы, добавляет вступление и пояснение после.
Разбор переживает это и даёт понятную ошибку, когда разобрать нельзя.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class JSONExtractionError(ValueError):
    """Ответ модели не удалось разобрать как JSON."""


def _candidates(text: str) -> list[str]:
    result = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    result.append(text.strip())
    return [c for c in result if c]


def _first_json_value(text: str) -> Any:
    """Ищет первое корректное JSON-значение (объект или массив) в произвольном тексте."""
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        return value
    raise JSONExtractionError("в ответе модели не найден JSON-объект")


def extract_json(text: str | None) -> Any:
    """Достаёт JSON из ответа модели: чистого, в заборах или с текстом до и после."""
    if text is None or not text.strip():
        raise JSONExtractionError("пустой ответ модели")
    cleaned = _THINK_RE.sub("", text).strip()
    if not cleaned:
        raise JSONExtractionError("ответ модели содержит только рассуждения, без JSON")
    for candidate in _candidates(cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return _first_json_value(candidate)
        except JSONExtractionError:
            continue
    preview = cleaned[:120].replace("\n", " ")
    raise JSONExtractionError(f"ответ модели не является JSON: «{preview}»")


def extract_json_object(text: str | None) -> dict[str, Any]:
    value = extract_json(text)
    if not isinstance(value, dict):
        raise JSONExtractionError(f"ожидался JSON-объект, получен {type(value).__name__}")
    return value
