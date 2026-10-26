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
from dataclasses import dataclass
from typing import Any

from app.agents import prompts
from app.core.logging import get_logger
from app.schemas.cards import InjectionAnswer

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
class InjectionVerdict:
    """Вердикт детектора: подозрителен ли фрагмент, почему и каким уровнем решено."""

    suspicious: bool
    reason: str = ""
    level: str = "regex"  # regex | llm | fallback


def _rule_reasons(text: str, metadata: dict[str, Any] | None = None) -> list[str]:
    reasons = [name for name, pattern in RULES if pattern.search(text)]
    if metadata and metadata.get("tiny_font") and reasons:
        reasons.append("текст мелким шрифтом")
    return reasons


def detect_injection_regex(text: str, metadata: dict[str, Any] | None = None) -> InjectionVerdict:
    """Быстрый уровень — бесплатные правила по тексту."""
    reasons = _rule_reasons(text, metadata)
    return InjectionVerdict(bool(reasons), reason="; ".join(reasons), level="regex")


async def detect_injection_llm(text: str) -> InjectionVerdict:
    """Точный уровень: узкий вопрос дешёвой модели через тот же клиент (таймаут, повторы,
    учёт стоимости). Исключения пробрасываются — решение «закрыть дверь» принимает вызывающий."""
    from app.services.structured import ask_model  # structured тянет клиента модели

    answer = await ask_model(
        prompts.injection_agent,
        prompts.build_injection_prompt(text),
        InjectionAnswer,
        max_retries=1,
    )
    return InjectionVerdict(answer.is_injection, reason=answer.reason, level="llm")


# Следы выполненной инъекции в готовом тексте: подозрительная цена «1 рубль» и т.п.
OUTPUT_RULES: list[tuple[str, re.Pattern[str]]] = [
    *RULES,
    (
        "подозрительная цена из инструкции",
        re.compile(r"(?<![\d.,])(?<!\d\s)\b1\s*(рубль|руб\.?|₽|rub)(?!\w)", re.IGNORECASE),
    ),
]


def output_traces(text: str) -> list[str]:
    """Следы служебных инструкций в сгенерированном тексте (для фильтра выхода)."""
    return [name for name, pattern in OUTPUT_RULES if pattern.search(text)]
