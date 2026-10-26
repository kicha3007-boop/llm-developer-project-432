"""Расчёт стоимости вызова модели. Деньги — Decimal, не float."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from app.core.config import Settings, settings

MILLION = Decimal(1_000_000)
QUANT = Decimal("0.00000001")  # совпадает с NUMERIC(14, 8) в таблице llm_calls


@dataclass(frozen=True)
class Price:
    input_per_1m: Decimal
    output_per_1m: Decimal


def price_for(tier: str, cfg: Settings = settings) -> Price:
    """Цена за миллион токенов по уровню модели: main — основная, cheap — служебная."""
    if tier == "cheap":
        return Price(cfg.llm_cheap_price_input_per_1m, cfg.llm_cheap_price_output_per_1m)
    if tier == "main":
        return Price(cfg.llm_price_input_per_1m, cfg.llm_price_output_per_1m)
    raise ValueError(f"Неизвестный уровень модели: {tier}")


def calculate_cost(price: Price, input_tokens: int, output_tokens: int) -> Decimal:
    """Стоимость из расхода токенов, который модель вернула вместе с ответом."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Расход токенов не может быть отрицательным")
    raw = (
        Decimal(input_tokens) * price.input_per_1m + Decimal(output_tokens) * price.output_per_1m
    ) / MILLION
    return raw.quantize(QUANT, rounding=ROUND_HALF_UP)


def cost_for_call(tier: str, input_tokens: int, output_tokens: int) -> Decimal:
    return calculate_cost(price_for(tier), input_tokens, output_tokens)
