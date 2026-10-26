"""Защита: персональные данные (``pii``), инъекции (``injection``) и фильтр выхода.

Фильтр выхода проверяет готовую карточку теми же детекторами: модель могла выполнить инъекцию
частично, и входные рубежи этого не увидят.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.guardrails.injection import output_traces
from app.guardrails.pii import PiiMasker, find_pii


@dataclass
class OutputCheck:
    clean: bool
    issues: list[str] = field(default_factory=list)
    card: dict[str, Any] = field(default_factory=dict)


def _strings(value: Any, path: str = ""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _strings(v, f"{path}[{i}]")


def _mask_card(value: Any, masker: PiiMasker) -> Any:
    if isinstance(value, str):
        return masker.mask(value)
    if isinstance(value, dict):
        return {k: _mask_card(v, masker) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_card(v, masker) for v in value]
    return value


def check_output(card: dict[str, Any]) -> OutputCheck:
    """Проверяет текстовые поля карточки на контакты, метки маскирования и следы инструкций.

    Найденные контакты маскируются в возвращаемой карточке, а карточка помечается нечистой —
    её дальше ведут к человеку, а не отдают как согласованную.
    """
    issues: list[str] = []
    for path, text in _strings({k: v for k, v in card.items() if k != "sources"}):
        kinds = find_pii(text)
        if kinds:
            issues.append(f"{path}: персональные данные ({', '.join(kinds)})")
        if "[ТЕЛЕФОН_" in text or "[EMAIL_" in text or "[ИНН_" in text or "[КАРТА_" in text:
            issues.append(f"{path}: метка маскирования попала в карточку")
        traces = output_traces(text)
        if traces:
            issues.append(f"{path}: следы служебных инструкций ({', '.join(traces)})")
    masked = _mask_card(card, PiiMasker()) if issues else card
    return OutputCheck(clean=not issues, issues=issues, card=masked)
