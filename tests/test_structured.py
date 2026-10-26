import pytest
from pydantic import ValidationError

from app.agents import prompts
from app.schemas.cards import TITLE_MAX_LEN, ProductCard
from app.services.structured import StructuredOutputError, generate_structured
from tests.conftest import as_json

CARD = {
    "title": "Погружной блендер МиксерПро 800, 800 Вт",
    "description": "Металлическая ножка, турбо-режим.",
    "characteristics": {"Мощность": "800 Вт"},
    "benefits": ["металлическая ножка"],
    "seo": {"meta_title": "Блендер МиксерПро 800", "meta_description": "Блендер 800 Вт"},
    "sources": [],
    "missing_fields": ["Цвет"],
    "confidence": 0.8,
}


def test_card_validators():
    with pytest.raises(ValidationError, match="title"):
        ProductCard.model_validate({**CARD, "title": "x" * (TITLE_MAX_LEN + 1)})
    with pytest.raises(ValidationError, match="confidence"):
        ProductCard.model_validate({**CARD, "confidence": 1.5})
    with pytest.raises(ValidationError, match="одновременно"):
        ProductCard.model_validate({**CARD, "missing_fields": ["мощность"]})
    card = ProductCard.model_validate(
        {**CARD, "characteristics": {"Мощность": "800 Вт", "Цвет": ""}}
    )
    assert card.characteristics == {"Мощность": "800 Вт"}


async def test_schema_passed_and_duplicated_in_prompt(fake_llm):
    fake_llm.add(as_json(CARD))
    card = await generate_structured(prompts.GENERATOR, "факты", ProductCard)
    assert card.title.startswith("Погружной")
    call = fake_llm.calls[0]
    assert call["schema"]["title"] == "ProductCard"
    assert "JSON-схеме" in call["prompt"]


async def test_invalid_json_is_repaired_with_error_in_prompt(fake_llm):
    fake_llm.add("Вот карточка: {битый json", as_json(CARD))
    card = await generate_structured(prompts.GENERATOR, "факты", ProductCard)
    assert card.confidence == 0.8
    assert len(fake_llm.calls) == 2
    repair_prompt = fake_llm.calls[1]["prompt"]
    assert "не прошёл проверку" in repair_prompt
    assert "не является JSON" in repair_prompt
    assert fake_llm.calls[1]["purpose"].endswith(":repair")


async def test_empty_answer_gives_error_after_limit(fake_llm):
    fake_llm.add("", "", "")
    with pytest.raises(StructuredOutputError, match="пустой ответ"):
        await generate_structured(prompts.GENERATOR, "факты", ProductCard, max_repairs=2)
    assert len(fake_llm.calls) == 3  # попыток ровно 1 + 2 лечения


async def test_long_title_is_repaired_pointwise(fake_llm):
    broken = {**CARD, "title": "Очень длинный заголовок " * 20}
    fake_llm.add(as_json(broken), as_json({"title": "Блендер МиксерПро 800"}))
    card = await generate_structured(prompts.GENERATOR, "факты", ProductCard)
    assert card.title == "Блендер МиксерПро 800"
    assert card.description == CARD["description"]  # описание не переделывалось
    repair = fake_llm.calls[1]
    assert repair["role"] == "field_repair"
    assert repair["purpose"].endswith("repair_field:title")
    assert "Текущий черновик" in repair["prompt"]


async def test_model_level_error_goes_to_full_repair(fake_llm):
    both = {**CARD, "missing_fields": ["Мощность"]}
    fake_llm.add(as_json(both), as_json(CARD))
    card = await generate_structured(prompts.GENERATOR, "факты", ProductCard)
    assert card.missing_fields == ["Цвет"]
    assert "одновременно" in fake_llm.calls[1]["prompt"]
