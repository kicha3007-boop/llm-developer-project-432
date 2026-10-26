import pytest
from pydantic import ValidationError

from app.schemas.cards import TITLE_MAX_LENGTH, CardDraft, Seo, SourceRef
from app.services import structured
from tests.conftest import as_json


def card(**overrides) -> dict:
    data = {
        "title": "МиксерПро 800 — блендер",
        "description": "Погружной блендер с металлической ножкой.",
        "characteristics": {"Мощность": "800 Вт"},
        "benefits": ["Металлическая ножка"],
        "seo": {"title": "Блендер МиксерПро 800", "description": "Блендер", "keywords": ["б"]},
        "sources": [{"chunk_id": "c1", "doc_id": "d1", "page": 3, "score": 0.9}],
        "missing_fields": ["Гарантия"],
        "confidence": 0.8,
    }
    data.update(overrides)
    return data


# --- контракт схемы ----------------------------------------------------------------------------


def test_title_limit_and_strip():
    assert TITLE_MAX_LENGTH == 60
    with pytest.raises(ValidationError, match="60"):
        CardDraft.model_validate(card(title="А" * 61))
    assert CardDraft.model_validate(card(title="  Блендер  ")).title == "Блендер"
    with pytest.raises(ValidationError):
        CardDraft.model_validate(card(title="   "))


def test_confidence_bounds():
    for bad in (1.5, -0.1):
        with pytest.raises(ValidationError):
            CardDraft.model_validate(card(confidence=bad))


def test_empty_characteristics_dropped():
    draft = CardDraft.model_validate(card(characteristics={"Мощность": "800 Вт", "Цвет": " "}))
    assert draft.characteristics == {"Мощность": "800 Вт"}


def test_field_not_in_both_lists():
    with pytest.raises(ValidationError, match="одновременно"):
        CardDraft.model_validate(card(missing_fields=["мощность"]))


def test_defaults():
    assert Seo().title == "" and Seo().keywords == []
    ref = SourceRef(chunk_id="c1")
    assert ref.doc_id == "" and ref.page is None and ref.score is None


# --- validate_or_retry -------------------------------------------------------------------------


async def test_first_try(fake_llm):
    fake_llm.add(as_json(card()))
    draft = await structured.validate_or_retry("Факты")
    assert draft.title.startswith("МиксерПро")
    assert fake_llm.roles() == ["Generator"]


async def test_garbage_is_repaired_with_error_in_prompt(fake_llm):
    fake_llm.add("Вот карточка: не JSON вообще", as_json(card()))
    draft = await structured.validate_or_retry("Факты")
    assert draft.confidence == 0.8
    assert "нет JSON" in fake_llm.calls[1]["prompt"]
    assert fake_llm.calls[1]["prompt"].startswith("Факты")  # исходная задача сохранена


async def test_long_title_fixed_pointwise_with_error_text(fake_llm):
    fake_llm.add(as_json(card(title="А" * 80)), as_json(card()))
    draft = await structured.validate_or_retry("Факты")
    assert draft.title == card()["title"]
    assert draft.description == card()["description"]
    repair = fake_llm.calls[1]
    assert repair["role"] == "FieldFixer"
    assert "60" in repair["prompt"] and "title" in repair["prompt"]


async def test_gives_up_after_limit(fake_llm):
    fake_llm.add(*[as_json(card(title="А" * 80))] * 3)
    with pytest.raises(structured.CardValidationError):
        await structured.validate_or_retry("Факты", max_retries=2)
    assert len(fake_llm.calls) == 3  # 1 попытка + 2 исправления


async def test_empty_answer_gives_error(fake_llm):
    fake_llm.add("", "", "")
    with pytest.raises(structured.CardValidationError, match="пустой ответ"):
        await structured.validate_or_retry("Факты")


async def test_model_level_error_goes_to_full_repair(fake_llm):
    fake_llm.add(as_json(card(missing_fields=["Мощность"])), as_json(card()))
    draft = await structured.validate_or_retry("Факты")
    assert draft.missing_fields == ["Гарантия"]
    assert fake_llm.roles() == ["Generator", "Generator"]
    assert "одновременно" in fake_llm.calls[1]["prompt"]


async def test_sdk_object_output_accepted(fake_llm):
    fake_llm.add(CardDraft.model_validate(card()))
    assert (await structured.validate_or_retry("Факты")).confidence == 0.8


# --- точечная переделка и порог ----------------------------------------------------------------


async def test_regenerate_field_changes_only_that_field(fake_llm):
    original = CardDraft.model_validate(card())
    fake_llm.role("FieldFixer", as_json(card(title="Короткий заголовок", description="другое")))
    result = await structured.regenerate_field(original, "title", "длиннее 60 символов")
    assert result.title == "Короткий заголовок"
    assert result.description == original.description
    assert "title" in fake_llm.calls[0]["prompt"]


def test_confidence_gate():
    assert structured.confidence_gate(CardDraft.model_validate(card(confidence=0.3))) == (
        "needs_review"
    )
    assert structured.confidence_gate(CardDraft.model_validate(card(confidence=0.87))) == "done"
