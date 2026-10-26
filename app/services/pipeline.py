"""Пайплайн генерации по тексту поставщика: извлечение → генерация → проверка с петлёй переделок.

Модуль ничего не знает про HTTP, Temporal и базу: его вызывают синхронный обработчик,
шаги устойчивого процесса и тесты. К модели — только через ``structured`` → ``llm.client``.
Петля переделки ограничена ``pipeline_max_rounds`` (3) кругами.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents import prompts
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cards import CritiqueReport, ProductCard, SupplierFacts
from app.schemas.jobs import JobStatus
from app.services.structured import generate_structured

log = get_logger(__name__)

MAX_ROUNDS = settings.pipeline_max_rounds
# Штраф уверенности за каждое недостающее поле: уверенность не может быть высокой,
# если важных данных нет, что бы ни написала модель.
MISSING_FIELD_PENALTY = 0.15
MIN_CONFIDENCE = 0.1


@dataclass
class PipelineResult:
    card: ProductCard | None
    attempts: int
    verdict: str  # approved | rejected — вердикт петли
    status: str  # итоговое состояние задачи: approved | needs_review
    issues: list[str] = field(default_factory=list)
    review_reason: str | None = None


def norm(text: str) -> str:
    """Нормализация для сравнения значений: регистр, ё, пробелы, запятая/точка, хвостовая точка."""
    text = text.lower().replace("ё", "е").replace(" ", " ").replace(",", ".")
    return " ".join(text.split()).rstrip(".").strip()


def cap_confidence(card: ProductCard) -> ProductCard:
    cap = max(MIN_CONFIDENCE, 1.0 - MISSING_FIELD_PENALTY * len(card.missing_fields))
    if card.confidence > cap:
        return card.model_copy(update={"confidence": round(cap, 2)})
    return card


def merge_missing(card: ProductCard, missing: list[str]) -> ProductCard:
    """Недостающие поля фактов переносятся в карточку, если модель их потеряла."""
    present = {norm(k) for k in card.characteristics}
    have = {norm(m) for m in card.missing_fields}
    extra = [m for m in missing if norm(m) not in present and norm(m) not in have]
    if not extra:
        return card
    return card.model_copy(update={"missing_fields": [*card.missing_fields, *extra]})


def code_checks(facts: SupplierFacts, card: ProductCard) -> list[str]:
    """Детерминированная часть проверки: значений, которых не было в фактах, быть не должно."""
    fact_values = {norm(v) for v in facts.characteristics.values()}
    issues = []
    for name, value in card.characteristics.items():
        if norm(value) not in fact_values and not any(norm(value) in fv for fv in fact_values):
            issues.append(f"characteristics.{name}: значения «{value}» нет в фактах")
    return issues


def decide(card: ProductCard, verdict: str, issues: list[str]) -> tuple[str, str | None]:
    """Итоговое состояние: низкая уверенность или неодобренный черновик — к человеку."""
    if verdict != "approved":
        return JobStatus.NEEDS_REVIEW, "проверка не пройдена за отведённые круги: " + "; ".join(
            issues[:5]
        )
    if card.confidence < settings.confidence_threshold:
        return JobStatus.NEEDS_REVIEW, (
            f"уверенность {card.confidence:.2f} ниже порога {settings.confidence_threshold:.2f}"
            + (f"; не хватает: {', '.join(card.missing_fields)}" if card.missing_fields else "")
        )
    return JobStatus.APPROVED, None


async def extract(supplier_text: str) -> SupplierFacts:
    log.info("step_extract_start")
    return await generate_structured(
        prompts.EXTRACTOR,
        prompts.build_extract_prompt(supplier_text),
        SupplierFacts,
        purpose="extract",
    )


async def generate(facts: SupplierFacts, feedback: list[str] | None = None) -> ProductCard:
    log.info("step_generate_start", feedback=bool(feedback))
    card = await generate_structured(
        prompts.GENERATOR,
        prompts.build_generate_prompt(facts.model_dump(), feedback),
        ProductCard,
        purpose="generate",
    )
    return cap_confidence(merge_missing(card, facts.missing_fields))


async def critique(facts: SupplierFacts, draft: ProductCard) -> CritiqueReport:
    log.info("step_critique_start")
    report = await generate_structured(
        prompts.CRITIC,
        prompts.build_critique_prompt(facts.model_dump(), draft.model_dump()),
        CritiqueReport,
        purpose="critique",
    )
    extra = code_checks(facts, draft)
    if extra:
        issues = [*report.issues, *({"field": "characteristics", "message": m} for m in extra)]
        return CritiqueReport.model_validate({"verdict": "regenerate", "issues": issues})
    return report


def issue_texts(report: CritiqueReport) -> list[str]:
    return [f"{i.field}: {i.message}" if i.field else i.message for i in report.issues]


async def run_pipeline(supplier_text: str, max_attempts: int = MAX_ROUNDS) -> PipelineResult:
    """Извлечение один раз, затем не больше ``max_attempts`` кругов «генерация — проверка»."""
    facts = await extract(supplier_text)
    feedback: list[str] = []
    attempts = 0
    draft: ProductCard | None = None
    verdict = "rejected"  # пессимистичное начальное значение: исчерпание кругов — отказ
    while attempts < max_attempts:
        attempts += 1
        draft = await generate(facts, feedback or None)
        report = await critique(facts, draft)
        if report.verdict == "approve":
            verdict = "approved"
            feedback = []
            break
        feedback = issue_texts(report)
        log.info("pipeline_regenerate", attempt=attempts, issues=feedback)
    assert draft is not None
    status, reason = decide(draft, verdict, feedback)
    return PipelineResult(
        card=draft,
        attempts=attempts,
        verdict=verdict,
        status=status,
        issues=feedback,
        review_reason=reason,
    )
