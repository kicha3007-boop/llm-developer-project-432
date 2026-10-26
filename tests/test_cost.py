from decimal import Decimal

import pytest

from app.llm.cost import Price, calculate_cost, price_for


def test_cost_from_token_usage_is_decimal():
    price = Price(Decimal("0.20"), Decimal("0.60"))
    cost = calculate_cost(price, input_tokens=1500, output_tokens=500)
    # 1500 * 0.20 / 1e6 + 500 * 0.60 / 1e6 = 0.0003 + 0.0003
    assert cost == Decimal("0.00060000")
    assert isinstance(cost, Decimal)


def test_no_float_drift_on_many_calls():
    price = Price(Decimal("0.1"), Decimal("0.2"))
    total = sum((calculate_cost(price, 1, 1) for _ in range(1000)), Decimal(0))
    assert total == Decimal("0.00030000")


def test_cheap_tier_is_cheaper():
    assert price_for("cheap").input_per_1m < price_for("main").input_per_1m


def test_negative_usage_rejected():
    with pytest.raises(ValueError):
        calculate_cost(Price(Decimal(1), Decimal(1)), -1, 0)


def test_unknown_tier():
    with pytest.raises(ValueError):
        price_for("gold")
