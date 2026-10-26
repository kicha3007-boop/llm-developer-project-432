"""Детекторы и маскирование персональных данных: телефон, почта, ИНН, номер карты.

Найденное заменяется метками вида «[PHONE_1]». Маскирование работает на фрагменте при сборке
контекста — до любого обращения к модели и до записи контекста в лог.

ИНН проверяется по контрольной сумме: детектор «по длине» маскировал бы артикулы, суммы и даты,
а защита, ломающая данные, — такой же отказ, как пропущенная атака. Номер карты — по алгоритму Луна.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

KIND_LABELS = {
    "phone": "PHONE",
    "email": "EMAIL",
    "inn": "INN",
    "card": "CARD",
}

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-])", re.UNICODE)
# +7 926 555-14-08, 8 (926) 555-14-08, +79265551408, 8-926-555-14-08
PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)
INN_RE = re.compile(r"(?<![\d\-])(\d{12}|\d{10})(?![\d\-])")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")

_INN10_WEIGHTS = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_1 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_2 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _checksum(digits: str, weights: tuple[int, ...]) -> int:
    return sum(int(d) * w for d, w in zip(digits, weights, strict=False)) % 11 % 10


def _inn_checksum(value: str) -> bool:
    """Контрольная сумма ИНН: 10 цифр — юрлицо, 12 цифр — физлицо/ИП."""
    if not value.isdigit():
        return False
    if len(value) == 10:
        return _checksum(value[:9], _INN10_WEIGHTS) == int(value[9])
    if len(value) == 12:
        return _checksum(value[:10], _INN12_WEIGHTS_1) == int(value[10]) and _checksum(
            value[:11], _INN12_WEIGHTS_2
        ) == int(value[11])
    return False


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class PiiFinding:
    kind: str
    label: str


@dataclass
class PiiMasker:
    """Маскировщик с общей нумерацией меток в пределах задачи: один телефон — одна метка."""

    _labels: dict[tuple[str, str], str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)
    findings: list[PiiFinding] = field(default_factory=list)

    def _label(self, kind: str, value: str) -> str:
        key = (kind, re.sub(r"\D", "", value) if kind != "email" else value.lower())
        if key not in self._labels:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            label = f"[{KIND_LABELS[kind]}_{self._counters[kind]}]"
            self._labels[key] = label
            self.findings.append(PiiFinding(kind=kind, label=label))
        return self._labels[key]

    def mask(self, text: str) -> str:
        text = EMAIL_RE.sub(lambda m: self._label("email", m.group(0)), text)
        text = PHONE_RE.sub(lambda m: self._label("phone", m.group(0)), text)

        def card(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19 and luhn_ok(digits):
                return self._label("card", m.group(0))
            return m.group(0)

        text = CARD_RE.sub(card, text)

        def inn(m: re.Match[str]) -> str:
            return self._label("inn", m.group(0)) if _inn_checksum(m.group(0)) else m.group(0)

        return INN_RE.sub(inn, text)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        return counts


def mask_pii(text: str, masker: PiiMasker | None = None) -> tuple[str, list[PiiFinding]]:
    """Маскирует персональные данные: (текст с метками, найденные сущности без исходных значений).

    Общий ``masker`` даёт сквозную нумерацию меток в пределах задачи.
    """
    masker = masker or PiiMasker()
    before = len(masker.findings)
    masked = masker.mask(text)
    return masked, masker.findings[before:]


def contains_pii(text: str) -> bool:
    return bool(find_pii(text))


def find_pii(text: str) -> list[str]:
    """Виды персональных данных, найденных в тексте (для фильтра выхода)."""
    masker = PiiMasker()
    masker.mask(text)
    return sorted({f.kind for f in masker.findings})


def mask_value(value: Any) -> Any:
    """Маскирование для логов: строки маскируются, вложенные структуры обходятся рекурсивно."""
    if isinstance(value, str):
        return PiiMasker().mask(value)
    if isinstance(value, dict):
        return {k: mask_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(mask_value(v) for v in value)
    return value
