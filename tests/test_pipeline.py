from app.schemas.jobs import JobStatus
from app.services import pipeline
from tests.conftest import as_json

FACTS = {
    "product_name": "Блендер МиксерПро 800",
    "characteristics": {"Мощность": "800 Вт", "Скоростей": "6"},
    "missing_fields": [],
}
CARD = {
    "title": "Погружной блендер МиксерПро 800",
    "description": "Блендер на 800 Вт с шестью скоростями.",
    "characteristics": {"Мощность": "800 Вт", "Скоростей": "6"},
    "benefits": ["шесть скоростей"],
    "missing_fields": [],
    "confidence": 0.9,
}
APPROVE = as_json({"verdict": "approve", "issues": []})
REGENERATE = as_json(
    {"verdict": "regenerate", "issues": [{"field": "title", "rule": "1", "message": "скучно"}]}
)


async def test_success_first_round(fake_llm):
    fake_llm.role("extractor", as_json(FACTS))
    fake_llm.role("generator", as_json(CARD))
    fake_llm.role("critic", APPROVE)
    result = await pipeline.run_pipeline("Блендер МиксерПро 800, 800 Вт, 6 скоростей")
    assert result.verdict == "approved"
    assert result.status == JobStatus.APPROVED
    assert result.attempts == 1
    assert fake_llm.roles() == ["extractor", "generator", "critic"]


async def test_regenerate_passes_feedback(fake_llm):
    fake_llm.role("extractor", as_json(FACTS))
    fake_llm.role("generator", as_json(CARD), as_json(CARD))
    fake_llm.role("critic", REGENERATE, APPROVE)
    result = await pipeline.run_pipeline("текст")
    assert result.attempts == 2
    assert result.verdict == "approved"
    second_generate = [c for c in fake_llm.calls if c["role"] == "generator"][1]
    assert "скучно" in second_generate["prompt"]
    assert fake_llm.roles().count("extractor") == 1  # извлечение вне петли


async def test_loop_is_bounded(fake_llm):
    fake_llm.role("extractor", as_json(FACTS))
    fake_llm.role("generator", as_json(CARD))
    fake_llm.role("critic", REGENERATE)
    result = await pipeline.run_pipeline("текст", max_attempts=3)
    assert result.attempts == 3
    assert result.verdict == "rejected"
    assert result.status == JobStatus.NEEDS_REVIEW
    assert result.card is not None  # черновик сохраняется и при отказе
    assert fake_llm.roles().count("generator") == 3


async def test_scant_data_gives_missing_fields_and_low_confidence(fake_llm):
    scant_facts = {
        "product_name": "Чайник",
        "characteristics": {"Мощность": "2200 Вт"},
        "missing_fields": ["Гарантия", "Цвет", "Страна-производитель"],
    }
    overconfident = {
        "title": "Чайник 2200 Вт",
        "characteristics": {"Мощность": "2200 Вт"},
        "missing_fields": [],  # модель «забыла» недостающие поля и завысила уверенность
        "confidence": 1.0,
    }
    fake_llm.role("extractor", as_json(scant_facts))
    fake_llm.role("generator", as_json(overconfident))
    fake_llm.role("critic", APPROVE)
    result = await pipeline.run_pipeline("Чайник 2200 Вт")
    assert result.card.missing_fields == ["Гарантия", "Цвет", "Страна-производитель"]
    assert result.card.confidence < 1.0
    assert result.status == JobStatus.NEEDS_REVIEW  # ждёт человека, а не «готово»
    assert "ниже порога" in result.review_reason


async def test_code_check_catches_invented_value(fake_llm):
    invented = {**CARD, "characteristics": {**CARD["characteristics"], "Вес": "5 кг"}}
    fake_llm.role("extractor", as_json(FACTS))
    fake_llm.role("generator", as_json(invented), as_json(CARD))
    fake_llm.role("critic", APPROVE)
    result = await pipeline.run_pipeline("текст")
    assert result.attempts == 2
    assert result.status == JobStatus.APPROVED
