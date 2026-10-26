"""Детектор инъекций через документы — в два уровня.

1. Быстрый уровень — правила по тексту: прямые приказы, маркеры служебных ролей, попытки вытащить
   промпт, длинные закодированные вставки, текст мелким шрифтом. Бесплатный, ловит грубое.
2. Точный уровень — узкий вопрос дешёвой модели со структурированным вердиктом. Он платный,
   поэтому идёт ТОЛЬКО по фрагментам, которые пометил первый уровень.

Если точный уровень недоступен (сеть, таймаут, невалидный ответ), фрагмент считается
подозрительным: защита, открывающая дверь при своём отказе, — не защита.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agents import prompts
from app.core.logging import get_logger
from app.schemas.cards import InjectionVerdict

log = get_logger(__name__)

RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "приказ игнорировать инструкции",
        re.compile(
            r"(игнорируй|игнорировать|забудь|не\s+учитывай|отмени)\s+(все\s+|всё\s+)?"
            r"(предыдущ|прошл|прежн|выше|данн|системн)\w*\s+(инструкц|указан|правил|промпт)",
            re.IGNORECASE,
        ),
    ),
    (
        "ignore previous instructions",
        re.compile(
            r"(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions|prompts?|rules)",
            re.IGNORECASE,
        ),
    ),
    (
        "маркер служебной роли",
        re.compile(
            r"(^|\n|\.\s)\s*(system|assistant|developer|система|ассистент)\s*:"
            r"|<\|?(im_start|im_end|system)\|?>|\[/?INST\]",
            re.IGNORECASE,
        ),
    ),
    (
        "попытка вытащить промпт",
        re.compile(
            r"(покажи|выведи|раскрой|повтори)\s+(свой\s+|свои\s+)?(системн\w+\s+)?"
            r"(промпт|инструкц)|reveal\s+(your\s+)?(system\s+)?prompt",
            re.IGNORECASE,
        ),
    ),
    (
        "смена роли",
        re.compile(r"(ты\s+теперь|отныне\s+ты|you\s+are\s+now|act\s+as)\b", re.IGNORECASE),
    ),
    (
        "приказ модели про содержимое карточки",
        re.compile(
            r"(при\s+генерации|в\s+карточк\w*|в\s+описани\w*|в\s+ответ\w*).{0,40}"
            r"(укажи|включи|добавь|напиши|вставь)",
            re.IGNORECASE,
        ),
    ),
    ("длинная закодированная вставка", re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")),
]


@dataclass
class Detection:
    suspicious: bool
    reasons: list[str] = field(default_factory=list)
    level: str = "rules"  # rules | model | fallback


def rule_check(text: str, meta: dict[str, Any] | None = None) -> list[str]:
    """Быстрый уровень: причины, по которым фрагмент подозрителен (пусто — чисто)."""
    reasons = [name for name, pattern in RULES if pattern.search(text)]
    if meta and meta.get("tiny_font") and reasons:
        reasons.append("текст мелким шрифтом")
    return reasons


async def model_check(text: str) -> InjectionVerdict:
    """Точный уровень: дешёвая модель через тот же клиент (таймаут, повторы, учёт стоимости)."""
    # импорт здесь, чтобы не тянуть structured при импорте модуля правил
    from app.services.structured import generate_structured

    return await generate_structured(
        prompts.INJECTION_JUDGE,
        prompts.build_injection_prompt(text),
        InjectionVerdict,
        purpose="injection_check",
        max_repairs=1,
    )


async def detect(text: str, meta: dict[str, Any] | None = None) -> Detection:
    reasons = rule_check(text, meta)
    if not reasons:
        return Detection(suspicious=False)
    try:
        verdict = await model_check(text)
    except Exception as exc:  # недоступный детектор не открывает дверь
        log.warning("injection_detector_unavailable", error=repr(exc))
        return Detection(
            suspicious=True,
            reasons=[*reasons, f"точный детектор недоступен ({type(exc).__name__})"],
            level="fallback",
        )
    if verdict.is_injection:
        return Detection(
            suspicious=True, reasons=[*reasons, f"модель: {verdict.reason}"], level="model"
        )
    return Detection(
        suspicious=False, reasons=[f"модель сняла подозрение: {verdict.reason}"], level="model"
    )


def output_traces(text: str) -> list[str]:
    """Следы служебных инструкций в сгенерированном тексте (для фильтра выхода)."""
    return [name for name, pattern in RULES if pattern.search(text)]
