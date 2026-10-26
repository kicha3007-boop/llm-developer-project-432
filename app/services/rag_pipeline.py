"""Сквозной сценарий: документы → поиск → контекст → генерация → проверка ссылок → карточка.

Три рубежа против выдумывания:
1. поиск — модель видит только найденные фрагменты;
2. проверка ссылок кодом — каждый источник обязан (а) существовать в базе и (б) входить в поданный
   контекст. Второе важнее: ссылка на существующий фрагмент, которого модель не видела, — это и есть
   выдумывание с правдоподобным видом. Нарушение — переделка с текстом ошибки, повтор — к человеку;
3. проверяющая роль сверяет утверждения карточки с заявленными источниками.

Функции модуля — это стадии; их вызывают и шаги процесса Temporal, и ``run_rag_pipeline``
(в процессе, для команды прогона метрик и тестов). Логика одна.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.agents import prompts
from app.core.config import settings
from app.core.logging import get_logger
from app.guardrails import check_output
from app.rag import retrieval
from app.rag.context import BuiltContext, ContextItem, SecurityReport, build_context
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.repositories.chunks import Hit
from app.schemas.cards import CritiqueReport, ProductCard, SourceRef
from app.schemas.jobs import JobStatus
from app.services.pipeline import cap_confidence, issue_texts, norm
from app.services.structured import generate_structured

log = get_logger(__name__)

# Запросы, которыми собираются сведения для карточки, кроме подсказки и заголовков документов
FIELD_QUERIES = (
    "технические характеристики параметры",
    "мощность объём вес размеры материал",
    "гарантия страна-производитель цвет",
    "описание преимущества комплект поставки",
)

StatusCallback = Callable[[str], None]


@dataclass
class SourceCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)
    card: ProductCard | None = None  # карточка с уточнёнными страницей и секцией источников


@dataclass
class RagResult:
    status: str
    attempts: int
    card: ProductCard | None
    review_reason: str | None
    security: SecurityReport
    context_chunk_ids: list[str]
    issues: list[str] = field(default_factory=list)

    def to_result(self) -> dict[str, Any]:
        return result_payload(
            self.card, self.security, self.context_chunk_ids, self.attempts, self.issues
        )


# --- Поиск -------------------------------------------------------------------------------------


def retrieval_queries(product_hint: str | None, titles: Sequence[str]) -> list[str]:
    queries: list[str] = []
    if product_hint:
        queries.append(product_hint)
    queries.extend(t for t in titles if t)
    base = product_hint or (titles[0] if titles else "")
    queries.extend(f"{base} {q}".strip() for q in FIELD_QUERIES)
    seen: set[str] = set()
    return [q for q in queries if not (q in seen or seen.add(q))]


def retrieve(document_ids: Sequence[str], product_hint: str | None) -> list[Hit]:
    """Гибридный поиск по подсказке, заголовкам документов и типовым полям карточки,
    с фильтром по выбранным документам. Повторы фрагментов схлопываются."""
    docs = documents_repo.get_many(list(document_ids))
    usable = [d["id"] for d in docs if d["status"] == "indexed"]
    titles = [d["title"] for d in docs if d["id"] in usable and d["title"]]
    hits: list[Hit] = []
    seen: set[str] = set()
    if usable:
        for query in retrieval_queries(product_hint, titles):
            for hit in retrieval.search(query, mode="hybrid", document_ids=usable):
                if hit.chunk_id not in seen:
                    seen.add(hit.chunk_id)
                    hits.append(hit)
    log.info("retrieval_done", documents=len(usable), hits=len(hits))
    return hits


# --- Генерация и проверка ----------------------------------------------------------------------


async def generate_from_context(
    context: BuiltContext, product_hint: str | None, feedback: list[str] | None = None
) -> ProductCard:
    log.info("step_generate_start", feedback=bool(feedback), chunks=len(context.items))
    card = await generate_structured(
        prompts.RAG_GENERATOR,
        prompts.build_rag_generate_prompt(context.text, context.chunk_ids, product_hint, feedback),
        ProductCard,
        purpose="rag_generate",
    )
    return cap_confidence(card)


def verify_sources(
    card: ProductCard, context_items: Sequence[ContextItem], existing_ids: set[str]
) -> SourceCheck:
    """Проверка ссылок по двум условиям: фрагмент есть в базе И был в поданном контексте.

    Заодно каждая характеристика обязана иметь источник, а страница и секция источника
    берутся из контекста, а не со слов модели.
    """
    by_id = {i.chunk_id: i for i in context_items}
    errors: list[str] = []
    fixed: list[SourceRef] = []
    for src in card.sources:
        if src.chunk_id not in existing_ids:
            errors.append(f"источник {src.chunk_id} не существует в базе — ссылка выдумана")
            continue
        if src.chunk_id not in by_id:
            errors.append(
                f"источник {src.chunk_id} не входил в поданный контекст — модель его не видела"
            )
            continue
        item = by_id[src.chunk_id]
        fixed.append(src.model_copy(update={"page": item.page, "section": item.section}))
    covered = {norm(s.field) for s in card.sources if s.field}
    for name in card.characteristics:
        if norm(name) not in covered:
            errors.append(f"у характеристики «{name}» нет источника в sources")
    if card.characteristics and not card.sources:
        errors.append("в карточке есть характеристики, но нет ни одного источника")
    return SourceCheck(
        ok=not errors, errors=errors, card=card.model_copy(update={"sources": fixed})
    )


def check_sources(card: ProductCard, context_items: Sequence[ContextItem]) -> SourceCheck:
    ids = [s.chunk_id for s in card.sources]
    return verify_sources(card, context_items, chunks_repo.existing_ids(ids))


async def critique_against_sources(context: BuiltContext, card: ProductCard) -> CritiqueReport:
    log.info("step_critique_start")
    return await generate_structured(
        prompts.RAG_CRITIC,
        prompts.build_rag_critique_prompt(context.text, card.model_dump()),
        CritiqueReport,
        purpose="rag_critique",
    )


# --- Решение и результат -----------------------------------------------------------------------


def result_payload(
    card: ProductCard | None,
    security: SecurityReport,
    context_ids: list[str],
    attempts: int,
    issues: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = card.model_dump() if card else {}
    payload["security_report"] = security.to_dict()
    payload["context_chunk_ids"] = context_ids
    payload["attempts"] = attempts
    if issues:
        payload["issues"] = issues
    return payload


def decide(
    card: ProductCard | None,
    *,
    verdict: str,
    issues: list[str],
    source_failures: int,
    security: SecurityReport,
) -> tuple[str, str | None, ProductCard | None]:
    """Итоговое состояние задачи после автоматических проверок."""
    reasons: list[str] = []
    if card is None:
        return (
            JobStatus.NEEDS_REVIEW,
            "карточка не собрана: в контексте нет пригодных фрагментов",
            None,
        )
    output = check_output(card.model_dump())
    if not output.clean:
        security.output_issues = output.issues
        card = ProductCard.model_validate(output.card)
        reasons.append("фильтр выхода: " + "; ".join(output.issues[:3]))
    if security.flagged_documents:
        reasons.append(
            "документ отправлен человеку: "
            + "; ".join(f"{d['document_id']} — {d['reason']}" for d in security.flagged_documents)
        )
    if source_failures:
        reasons.append("ссылки на источники не прошли проверку: " + "; ".join(issues[:3]))
    elif verdict != "approved":
        reasons.append("проверка не пройдена за отведённые круги: " + "; ".join(issues[:3]))
    if card.confidence < settings.confidence_threshold:
        reasons.append(
            f"уверенность {card.confidence:.2f} ниже порога {settings.confidence_threshold:.2f}"
        )
    if reasons:
        return JobStatus.NEEDS_REVIEW, "; ".join(reasons), card
    return JobStatus.APPROVED, None, card


async def run_rag_pipeline(
    document_ids: Sequence[str],
    product_hint: str | None = None,
    *,
    on_status: StatusCallback | None = None,
    max_rounds: int | None = None,
    hits: Sequence[Hit] | None = None,
) -> RagResult:
    """Весь сценарий в одном процессе. В Temporal те же стадии идут отдельными шагами."""
    max_rounds = max_rounds or settings.pipeline_max_rounds

    def status(value: str) -> None:
        if on_status:
            on_status(value)

    status(JobStatus.RETRIEVING)
    found = list(hits) if hits is not None else retrieve(document_ids, product_hint)
    context = await build_context(found)
    if not context.items:
        return RagResult(
            status=JobStatus.NEEDS_REVIEW,
            attempts=0,
            card=None,
            review_reason="в контексте нет пригодных фрагментов (поиск пуст или всё исключено)",
            security=context.security,
            context_chunk_ids=[],
        )

    feedback: list[str] = []
    verdict = "rejected"
    source_failures = 0
    attempts = 0
    card: ProductCard | None = None
    while attempts < max_rounds:
        attempts += 1
        status(JobStatus.GENERATING if attempts == 1 else JobStatus.REGENERATING)
        card = await generate_from_context(context, product_hint, feedback or None)
        status(JobStatus.VERIFYING)
        check = check_sources(card, context.items)
        card = check.card or card
        if not check.ok:
            source_failures += 1
            feedback = check.errors
            log.warning("sources_rejected", attempt=attempts, errors=check.errors)
            if source_failures >= 2:  # повторное нарушение — к человеку
                break
            continue
        source_failures = 0
        report = await critique_against_sources(context, card)
        if report.verdict == "approve":
            verdict = "approved"
            feedback = []
            break
        feedback = issue_texts(report)
        log.info("pipeline_regenerate", attempt=attempts, issues=feedback)

    final_status, reason, card = decide(
        card,
        verdict=verdict,
        issues=feedback,
        source_failures=source_failures,
        security=context.security,
    )
    log.info("rag_pipeline_done", status=final_status, attempts=attempts, reason=reason)
    return RagResult(
        status=final_status,
        attempts=attempts,
        card=card,
        review_reason=reason,
        security=context.security,
        context_chunk_ids=context.chunk_ids,
        issues=feedback,
    )
