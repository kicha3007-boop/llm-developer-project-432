"""Метрики: нормализация значений, цитирование, промпт судьи на подмене."""

import json

from app.core.config import settings
from evals.metrics import (
    characteristics_score,
    citation_score,
    judge_card,
    normalize_value,
    values_match,
    weakest_metric,
)
from evals.run_evals import averages, render_table
from tests.conftest import DATA, as_json


def test_normalization_makes_units_equal():
    assert normalize_value("800 Вт") == normalize_value("800 вт") == normalize_value("800 Вт.")
    assert values_match("1,5 л", "1.5 л")
    assert values_match("220 В, 50 Гц", "220В, 50Гц")
    assert not values_match("800 Вт", "1200 Вт")


def test_characteristics_score_on_golden_shape():
    golden = json.loads((DATA / "golden_cards.json").read_text(encoding="utf-8"))
    expected = golden["documents"]["kettle_manual.pdf"]["characteristics"]
    card = {
        "characteristics": {
            "мощность": "2200 вт",
            "Объём": "1.7 л",
            "Артикул": "KTL-1700",
            "Модель": "КеттлПро 1.7",
            "Гарантия": "12 месяцев.",
        }
    }
    score = characteristics_score(expected, card)
    assert score.matched == 5
    assert score.total == len(expected)


def test_characteristics_score_for_price_rows():
    expected = {"BLD-800": "МиксерПро 800, 800 Вт, графитовый, 2190 ₽"}
    card = {
        "title": "МиксерПро 800",
        "characteristics": {"Мощность": "800 Вт", "Цвет": "графитовый", "Цена": "2 190 ₽"},
    }
    assert characteristics_score(expected, card).score == 1.0


def test_citation_score_checks_chunk_content_and_probes():
    card = {
        "characteristics": {"Мощность": "800 Вт", "Гарантия": "24 месяца"},
        "sources": [
            {"chunk_id": "a", "field": "Мощность"},
            {"chunk_id": "b", "field": "Гарантия"},  # фрагмент «b» гарантии не содержит
        ],
    }
    chunks = {"a": {"text": "Мощность: 800 Вт", "page": 1}, "b": {"text": "Цвет", "page": 2}}
    probes = {
        "Мощность": {"page": 1, "text_contains": "800 Вт"},
        "Гарантия": {"page": 1, "text_contains": "24 месяца"},
    }
    score = citation_score(card, chunks, probes)
    assert (score.supported, score.total) == (1, 2)
    assert (score.probes_ok, score.probes_total) == (1, 2)


async def test_judge_prompt_on_fake_model(fake_llm):
    fake_llm.role(
        "Judge",
        as_json({"supported": 3, "total": 4, "unsupported_claims": ["цвет графитовый"]}),
    )
    verdict = await judge_card({"title": "Блендер"}, "[a | f.pdf | стр. 1 | Х]\nМощность: 800 Вт")
    assert verdict.score == 0.75
    call = fake_llm.calls[0]
    assert call["model"] == settings.llm_cheap_model
    assert "Мощность: 800 Вт" in call["prompt"] and "Блендер" in call["prompt"]


def test_report_table_and_weakest():
    rows = [
        {
            "document": "a.pdf",
            "status": "approved",
            "characteristics": 1.0,
            "citation": 0.5,
            "faithfulness": 0.9,
            "cost": "0.001",
        },
        {"document": "b.pdf", "status": "failed"},
    ]
    avg = averages(rows)
    assert weakest_metric(avg) == ("citation", 0.5)
    table = render_table(rows, avg)
    assert "a.pdf" in table and "среднее" in table
