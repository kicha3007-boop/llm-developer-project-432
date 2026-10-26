"""Метрики качества генерации на эталоне ``data/golden_cards.json``.

1. Точность характеристик — доля эталонных характеристик, совпавших с карточкой. Сравнение не
   буквальное: «800 Вт», «800 вт» и «800 Вт.» — одно и то же; нормализация — часть метрики.
2. Цитирование — доля источников карточки, чей фрагмент действительно содержит значение
   характеристики, на которую ссылается (+ проверка «зондов» эталона: страница и текст).
3. Подтверждённость — вердикт модели-судьи: сколько утверждений карточки подтверждают фрагменты.
   Модель в роли судьи — оценка, а не истина.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agents import prompts
from app.schemas.cards import JudgeVerdict
from app.services.structured import ask_model

_TRAILING = re.compile(r"[.;:]+$")


def normalize_value(value: Any) -> str:
    """Нормализация значения: регистр, ё, пробелы, десятичная запятая, точка в конце."""
    text = str(value).lower().replace("ё", "е").replace(" ", " ")
    text = text.replace(",", ".")
    text = re.sub(r"\s+", "", text)
    return _TRAILING.sub("", text)


def normalize_key(key: str) -> str:
    return re.sub(r"[\s\-_]+", " ", str(key).lower().replace("ё", "е")).strip(" .:")


def values_match(expected: str, actual: str) -> bool:
    e, a = normalize_value(expected), normalize_value(actual)
    return bool(e) and (e == a or e in a or (len(a) >= 3 and a in e))


def _find_key(expected_key: str, actual: dict[str, str]) -> str | None:
    ek = normalize_key(expected_key)
    for key in actual:
        if normalize_key(key) == ek:
            return key
    for key in actual:
        nk = normalize_key(key)
        if ek in nk or nk in ek:
            return key
    return None


@dataclass
class FieldResult:
    field: str
    expected: str
    actual: str | None
    ok: bool


@dataclass
class CharacteristicsScore:
    matched: int
    total: int
    details: list[FieldResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.matched / self.total if self.total else 1.0


def flatten_card(card: dict[str, Any]) -> str:
    parts = [card.get("title", ""), card.get("description", "")]
    parts += [f"{k} {v}" for k, v in (card.get("characteristics") or {}).items()]
    parts += list(card.get("benefits") or [])
    return " ".join(str(p) for p in parts)


def characteristics_score(expected: dict[str, str], card: dict[str, Any]) -> CharacteristicsScore:
    actual: dict[str, str] = card.get("characteristics") or {}
    flat = normalize_value(flatten_card(card))
    details = []
    for key, exp in expected.items():
        found = _find_key(key, actual)
        if found is not None:
            ok = values_match(exp, actual[found])
            details.append(FieldResult(key, exp, actual[found], ok))
            continue
        # строки КП/спецификации: «МиксерПро 800, 800 Вт, графитовый, 2190 ₽» —
        # засчитываем, если все части значения есть в тексте карточки
        parts = [p for p in (normalize_value(x) for x in exp.split(",")) if p]
        ok = bool(parts) and all(p in flat for p in parts)
        details.append(FieldResult(key, exp, None, ok))
    return CharacteristicsScore(sum(d.ok for d in details), len(details), details)


@dataclass
class CitationScore:
    supported: int
    total: int
    probes_ok: int
    probes_total: int
    problems: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.supported / self.total if self.total else 0.0

    @property
    def probe_score(self) -> float:
        return self.probes_ok / self.probes_total if self.probes_total else 1.0


def citation_score(
    card: dict[str, Any],
    chunks: dict[str, dict[str, Any]],
    probes: dict[str, dict[str, Any]] | None = None,
) -> CitationScore:
    """``chunks``: chunk_id → {"text", "page"}. Источник засчитан, если фрагмент
    содержит значение характеристики."""
    characteristics: dict[str, str] = card.get("characteristics") or {}
    supported = total = 0
    problems: list[str] = []
    for src in card.get("sources") or []:
        name = src.get("field")
        key = _find_key(name, characteristics) if name else None
        if key is None:
            continue
        total += 1
        chunk = chunks.get(src.get("chunk_id"))
        if chunk and normalize_value(characteristics[key]) in normalize_value(chunk["text"]):
            supported += 1
        else:
            problems.append(f"{key}: фрагмент {src.get('chunk_id')} не содержит значения")
    probes_ok = probes_total = 0
    for name, probe in (probes or {}).items():
        probes_total += 1
        ids = [
            s.get("chunk_id")
            for s in card.get("sources") or []
            if s.get("field") and normalize_key(s["field"]) == normalize_key(name)
        ]
        ok = False
        for cid in ids:
            chunk = chunks.get(cid)
            if not chunk:
                continue
            page_ok = probe.get("page") is None or chunk.get("page") == probe["page"]
            text_ok = normalize_value(probe["text_contains"]) in normalize_value(chunk["text"])
            ok = ok or (page_ok and text_ok)
        if not ok:
            problems.append(f"зонд «{name}»: нет источника на стр. {probe.get('page')}")
        probes_ok += ok
    return CitationScore(supported, total, probes_ok, probes_total, problems)


async def judge_card(card: dict[str, Any], context_text: str) -> JudgeVerdict:
    """Третья метрика: модель-судья (дешёвая) получает карточку и фрагменты, выносит вердикт."""
    return await ask_model(
        prompts.judge_agent,
        prompts.build_judge_prompt(card, context_text),
        JudgeVerdict,
        max_retries=1,
    )


def weakest_metric(averages: dict[str, float]) -> tuple[str, float]:
    name = min(averages, key=lambda k: averages[k])
    return name, averages[name]
