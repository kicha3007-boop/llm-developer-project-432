"""Защита на границах: вход (``guard_context``) и выход (``guard_output``).

``guard_context(chunks)`` — фрагменты документов перед сборкой контекста:
1. персональные данные маскируются (``pii``) — до любого обращения к модели и до логов;
2. правила (``detect_injection_regex``) помечают подозрительные фрагменты, и только их
   проверяет дешёвая модель (``detect_injection_llm``). Подтверждено или детектор недоступен —
   фрагмент выбрасывается (fail-closed). Слишком много подозрительных — ``needs_review``.

``guard_output(card_json)`` — готовая карточка теми же детекторами: модель могла выполнить
инъекцию частично, и входные рубежи этого не увидят.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.guardrails.injection import (
    InjectionVerdict,
    detect_injection_llm,
    detect_injection_regex,
    output_traces,
)
from app.guardrails.pii import PiiMasker, find_pii, mask_pii

__all__ = [
    "GuardReport",
    "InjectionVerdict",
    "detect_injection_llm",
    "detect_injection_regex",
    "guard_context",
    "guard_output",
]

log = get_logger(__name__)


@dataclass
class GuardReport:
    """Отчёт защиты для результата задачи: что замаскировано, что исключено и почему."""

    masked_pii: int = 0
    masked_kinds: dict[str, int] = field(default_factory=dict)
    suspicious_chunks: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    output_issues: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return len(self.suspicious_chunks) > settings.suspicious_chunk_limit

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "needs_review": self.needs_review}


async def guard_context(
    chunks: list[dict[str, Any]], masker: PiiMasker | None = None
) -> tuple[list[dict[str, Any]], GuardReport]:
    """Фрагменты ``{"chunk_id", "text", "metadata", ...}`` → (безопасные фрагменты, отчёт)."""
    masker = masker or PiiMasker()
    report = GuardReport()
    safe: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") or chunk.get("id")
        text, entities = mask_pii(chunk.get("text", ""), masker)
        report.masked_pii += len(entities)
        for entity in entities:
            report.masked_kinds[entity.kind] = report.masked_kinds.get(entity.kind, 0) + 1
        verdict = detect_injection_regex(text, chunk.get("metadata"))
        if verdict.suspicious:
            try:
                confirmed = await detect_injection_llm(text)
                suspicious, reason = confirmed.suspicious, confirmed.reason
            except Exception as exc:  # недоступный детектор не открывает дверь
                log.warning("injection_detector_unavailable", error=repr(exc))
                suspicious = True
                reason = f"{verdict.reason}; точный детектор недоступен ({type(exc).__name__})"
            if suspicious:
                report.suspicious_chunks.append(chunk_id)
                report.reasons[chunk_id] = reason or verdict.reason
                log.warning("context_chunk_excluded", chunk_id=chunk_id, reason=reason)
                continue
        safe.append({**chunk, "text": text})
    log.info(
        "guard_context_done",
        chunks=len(chunks),
        kept=len(safe),
        masked=report.masked_kinds,
        suspicious=report.suspicious_chunks,
    )
    return safe, report


def _strings(value: Any, path: str = ""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _strings(v, f"{path}[{i}]")


def _mask_deep(value: Any, masker: PiiMasker) -> Any:
    if isinstance(value, str):
        return masker.mask(value)
    if isinstance(value, dict):
        return {k: _mask_deep(v, masker) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_deep(v, masker) for v in value]
    return value


def guard_output(card_json: str) -> tuple[str, list[str]]:
    """Фильтр выхода: (карточка JSON с замаскированными контактами, список замечаний).

    Замечания: персональные данные, метки маскирования и следы служебных инструкций
    (в том числе «цена 1 рубль») в текстовых полях. Непустой список — карточка к человеку.
    """
    card = json.loads(card_json)
    issues: list[str] = []
    checked = {k: v for k, v in card.items() if k != "sources"} if isinstance(card, dict) else card
    for path, text in _strings(checked):
        kinds = find_pii(text)
        if kinds:
            issues.append(f"{path}: персональные данные ({', '.join(kinds)})")
        if any(f"[{label}_" in text for label in ("PHONE", "EMAIL", "INN", "CARD")):
            issues.append(f"{path}: метка маскирования попала в карточку")
        traces = output_traces(text)
        if traces:
            issues.append(f"{path}: следы служебных инструкций ({', '.join(traces)})")
    cleaned = _mask_deep(card, PiiMasker()) if issues else card
    return json.dumps(cleaned, ensure_ascii=False), issues
