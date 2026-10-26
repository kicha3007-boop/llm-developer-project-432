"""Пайплайн по тексту поставщика: извлечение → генерация → проверка с петлёй переделок.

Как в эталоне курса (урок «Разбор примера», ``services/steps.py`` → здесь ``pipeline.py``):
стадии ``extract`` / ``generate`` / ``critique`` и ``run_pipeline``, который возвращает
``(черновик, число попыток, вердикт, статус)``. Модуль ничего не знает про HTTP, Temporal и базу.
Петля ограничена ``max_pipeline_rounds`` (3) кругами. Статус: ``done`` — критик одобрил и
уверенность не ниже порога, ``needs_review`` — ниже порога, ``rejected`` — круги исчерпаны.
"""

from __future__ import annotations

from app.agents import prompts
from app.agents.prompts import critic_agent, extractor_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import run_agent
from app.schemas.cards import CardDraft, CritiqueReport, SupplierFacts
from app.schemas.jobs import JobStatus
from app.services import structured

log = get_logger(__name__)

# Штраф уверенности за каждое недостающее поле: уверенность не бывает высокой, если важных
# данных нет, что бы ни написала модель.
MISSING_FIELD_PENALTY = 0.15
MIN_CONFIDENCE = 0.1


def norm(text: str) -> str:
    """Нормализация для сравнения значений: регистр, ё, пробелы, десятичная запятая, точка."""
    text = text.lower().replace("ё", "е").replace(" ", " ").replace(",", ".")
    return " ".join(text.split()).rstrip(".").strip()


def cap_confidence(card: CardDraft) -> CardDraft:
    cap = max(MIN_CONFIDENCE, 1.0 - MISSING_FIELD_PENALTY * len(card.missing_fields))
    if card.confidence > cap:
        return card.model_copy(update={"confidence": round(cap, 2)})
    return card


def merge_missing(card: CardDraft, missing: list[str]) -> CardDraft:
    """Недостающие поля фактов переносятся в карточку, если модель их потеряла."""
    present = {norm(k) for k in card.characteristics}
    have = {norm(m) for m in card.missing_fields}
    extra = [m for m in missing if norm(m) not in present and norm(m) not in have]
    if not extra:
        return card
    return card.model_copy(update={"missing_fields": [*card.missing_fields, *extra]})


def code_checks(facts: SupplierFacts, card: CardDraft) -> list[str]:
    """Детерминированная часть проверки: значений, которых нет в фактах, быть не должно."""
    values = {norm(v) for v in facts.characteristics.values()}
    return [
        f"characteristics.{name}: значения «{value}» нет в фактах"
        for name, value in card.characteristics.items()
        if norm(value) not in values and not any(norm(value) in v for v in values)
    ]


def decide(card: CardDraft, verdict: str) -> str:
    if verdict != "approved":
        return JobStatus.REJECTED
    return structured.confidence_gate(card)


async def extract(supplier_text: str) -> SupplierFacts:
    log.info("step_extract_start")
    return await structured.ask_model(
        extractor_agent,
        prompts.build_extract_prompt(supplier_text),
        SupplierFacts,
        runner=run_agent,
    )


async def generate(facts: SupplierFacts, feedback: list[str] | None = None) -> CardDraft:
    log.info("step_generate_start", feedback=bool(feedback))
    card = await structured.validate_or_retry(prompts.build_generate_prompt(facts, feedback))
    return cap_confidence(merge_missing(card, facts.missing_fields))


async def critique(facts: SupplierFacts, draft: CardDraft) -> CritiqueReport:
    log.info("step_critique_start")
    report = await structured.ask_model(
        critic_agent, prompts.build_critique_prompt(facts, draft), CritiqueReport, runner=run_agent
    )
    extra = code_checks(facts, draft)
    if extra:
        return CritiqueReport(verdict="regenerate", issues=[*report.issues, *extra])
    return report


async def run_pipeline(
    supplier_text: str, max_attempts: int | None = None
) -> tuple[CardDraft, int, str, str]:
    """Извлечение один раз, затем не больше ``max_attempts`` кругов «генерация — проверка»."""
    max_attempts = max_attempts or settings.max_pipeline_rounds
    facts = await extract(supplier_text)
    feedback: list[str] = []
    attempts = 0
    draft: CardDraft | None = None
    verdict = "rejected"  # пессимистичное начальное значение: исчерпание кругов — отказ
    while attempts < max_attempts:
        attempts += 1
        draft = await generate(facts, feedback or None)
        report = await critique(facts, draft)
        if report.verdict == "approve":
            verdict = "approved"
            break
        feedback = report.issues
        log.info("pipeline_regenerate", attempt=attempts, issues=feedback)
    assert draft is not None
    return draft, attempts, verdict, decide(draft, verdict)
