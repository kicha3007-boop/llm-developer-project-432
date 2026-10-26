from app.schemas.cards import CardDraft, CritiqueReport, SupplierFacts
from app.services import pipeline
from tests.conftest import as_json


def facts(**kw) -> SupplierFacts:
    data = {
        "product_name": "Блендер МиксерПро 800",
        "characteristics": {"Мощность": "800 Вт", "Скоростей": "6"},
    }
    return SupplierFacts(**{**data, **kw})


def draft(**kw) -> CardDraft:
    data = {
        "title": "Погружной блендер МиксерПро 800",
        "description": "Блендер на 800 Вт с шестью скоростями.",
        "characteristics": {"Мощность": "800 Вт"},
        "benefits": ["шесть скоростей"],
        "confidence": 0.9,
    }
    return CardDraft(**{**data, **kw})


APPROVE = CritiqueReport(verdict="approve").model_dump_json()
REGENERATE = CritiqueReport(verdict="regenerate", issues=["заголовок скучный"]).model_dump_json()


async def test_extract_parses_fenced_json(fake_llm):
    fake_llm.role("Extractor", "```json\n" + facts().model_dump_json() + "\n```")
    result = await pipeline.extract("текст")
    assert result.characteristics["Мощность"] == "800 Вт"


async def test_success_first_round(fake_llm):
    fake_llm.add(facts().model_dump_json(), draft().model_dump_json(), APPROVE)
    result, attempts, verdict, status = await pipeline.run_pipeline("текст")
    assert (verdict, status, attempts) == ("approved", "done", 1)
    assert result.title.startswith("Погружной")
    assert fake_llm.roles() == ["Extractor", "Generator", "Critic"]


async def test_regenerate_passes_feedback(fake_llm):
    fake_llm.role("Extractor", facts().model_dump_json())
    fake_llm.role("Generator", draft().model_dump_json())
    fake_llm.role("Critic", REGENERATE, APPROVE)
    _d, attempts, verdict, status = await pipeline.run_pipeline("текст")
    assert (attempts, verdict, status) == (2, "approved", "done")
    second = [c for c in fake_llm.calls if c["role"] == "Generator"][1]
    assert "заголовок скучный" in second["prompt"]
    assert fake_llm.roles().count("Extractor") == 1  # извлечение вне петли


async def test_loop_is_bounded(fake_llm):
    fake_llm.role("Extractor", facts().model_dump_json())
    fake_llm.role("Generator", draft().model_dump_json())
    fake_llm.role("Critic", REGENERATE)
    result, attempts, verdict, status = await pipeline.run_pipeline("текст", max_attempts=3)
    assert (attempts, verdict, status) == (3, "rejected", "rejected")
    assert result is not None  # черновик сохраняется и при отказе
    assert fake_llm.roles().count("Generator") == 3


async def test_scant_data_low_confidence_needs_review(fake_llm):
    fake_llm.role(
        "Extractor",
        facts(
            characteristics={"Мощность": "2200 Вт"},
            missing_fields=["Гарантия", "Цвет", "Страна-производитель"],
        ).model_dump_json(),
    )
    # модель «забыла» недостающие поля и завысила уверенность
    fake_llm.role(
        "Generator",
        as_json(
            {
                "title": "Чайник 2200 Вт",
                "characteristics": {"Мощность": "2200 Вт"},
                "confidence": 1.0,
            }
        ),
    )
    fake_llm.role("Critic", APPROVE)
    result, _a, verdict, status = await pipeline.run_pipeline("Чайник 2200 Вт")
    assert result.missing_fields == ["Гарантия", "Цвет", "Страна-производитель"]
    assert result.confidence < 1.0
    assert (verdict, status) == ("approved", "needs_review")


async def test_code_check_catches_invented_value(fake_llm):
    fake_llm.role("Extractor", facts().model_dump_json())
    invented = draft(characteristics={"Мощность": "800 Вт", "Вес": "5 кг"})
    fake_llm.role("Generator", invented.model_dump_json(), draft().model_dump_json())
    fake_llm.role("Critic", APPROVE)
    _d, attempts, _v, status = await pipeline.run_pipeline("текст")
    assert (attempts, status) == (2, "done")


def test_critic_runs_on_cheap_model():
    from app.agents.prompts import critic_agent, generator_agent
    from app.core.config import settings

    assert critic_agent.model == settings.llm_cheap_model
    assert generator_agent.model == settings.llm_model
