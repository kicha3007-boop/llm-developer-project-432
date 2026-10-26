"""Шаги процессов Temporal — тонкие обёртки над сервисами.

Единица восстановления в устойчивом исполнении — шаг: выполненный шаг записан в историю и после
рестарта воркера заново не выполняется. Поэтому стадии — отдельные шаги, а не один большой.

Шаги, которые не отдают управление (база, диск, разбор документа, эмбеддинги), объявлены
синхронными и исполняются в пуле потоков воркера. Шаги с вызовом модели — асинхронные.
Каждый шаг привязывает ``job_id`` к контексту логов: поиск по нему показывает весь путь задачи.
Данные между шагами — JSON-совместимые словари и строки.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from temporalio import activity

from app.core.db import connection
from app.core.logging import bind_context, get_logger
from app.guardrails import GuardReport
from app.rag.context import context_chunk_ids
from app.repositories import jobs as jobs_repo
from app.schemas.cards import CardDraft, CritiqueReport, SupplierFacts
from app.services import documents as documents_service
from app.services import pipeline, rag_pipeline

log = get_logger(__name__)


def _bind(job_id: str | None = None, **extra: Any) -> None:
    info = activity.info()
    bind_context(
        job_id=job_id,
        workflow_id=info.workflow_id,
        activity=info.activity_type,
        activity_attempt=info.attempt,
        **extra,
    )


def guard_from_dict(data: dict[str, Any]) -> GuardReport:
    names = {f.name for f in fields(GuardReport)}
    return GuardReport(**{k: v for k, v in data.items() if k in names})


def critique_verdict(report: dict[str, Any]) -> tuple[bool, list[str]]:
    parsed = CritiqueReport.model_validate(report)
    return parsed.verdict == "approve", parsed.issues


# --- Состояние задачи --------------------------------------------------------------------------


@activity.defn
def set_job_status(
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    review_reason: str | None = None,
    bump_attempts: bool = False,
) -> None:
    """Статус в базе пишет отдельный короткий шаг, а не пайплайн изнутри."""
    _bind(job_id)
    with connection() as conn:
        jobs_repo.set_status(
            conn,
            job_id,
            status,
            result=result,
            error=error,
            review_reason=review_reason,
            bump_attempts=bump_attempts,
        )
    log.info("job_status", status=status)


# --- Пайплайн по тексту ------------------------------------------------------------------------


@activity.defn
async def extract_activity(job_id: str, supplier_text: str) -> dict[str, Any]:
    _bind(job_id)
    return (await pipeline.extract(supplier_text)).model_dump()


@activity.defn
async def generate_activity(
    job_id: str, facts: dict[str, Any], feedback: list[str] | None = None
) -> dict[str, Any]:
    _bind(job_id)
    card = await pipeline.generate(SupplierFacts.model_validate(facts), feedback)
    return card.model_dump()


@activity.defn
async def critique_activity(
    job_id: str, facts: dict[str, Any], card: dict[str, Any]
) -> dict[str, Any]:
    _bind(job_id)
    report = await pipeline.critique(
        SupplierFacts.model_validate(facts), CardDraft.model_validate(card)
    )
    return report.model_dump()


@activity.defn
def finalize_text_activity(
    job_id: str, card: dict[str, Any], verdict: str, issues: list[str], attempts: int
) -> dict[str, Any]:
    """Решение по пайплайну из текста и запись результата."""
    _bind(job_id)
    draft = CardDraft.model_validate(card)
    status = pipeline.decide(draft, verdict)
    reason = None
    if status == "needs_review":
        reason = f"уверенность {draft.confidence:.2f} ниже порога"
    elif status == "rejected":
        reason = "критик не одобрил за отведённые круги: " + "; ".join(issues[:3])
    result = {**draft.model_dump(), "attempts": attempts, "verdict": verdict}
    with connection() as conn:
        jobs_repo.set_status(conn, job_id, status, result=result, review_reason=reason)
    log.info("job_finalized", status=status, reason=reason)
    return {"status": status, "review_reason": reason}


# --- Документы ---------------------------------------------------------------------------------


@activity.defn
def parse_document_activity(document_id: str, job_id: str | None = None) -> dict[str, Any]:
    _bind(job_id, document_id=document_id)
    return documents_service.parse_document(document_id)


@activity.defn
def index_document_activity(document_id: str, job_id: str | None = None) -> dict[str, Any]:
    _bind(job_id, document_id=document_id)
    embedded = documents_service.index_document(document_id)
    return {
        "document_id": document_id,
        "status": documents_service.mark_indexed(document_id),
        "embedded": embedded,
    }


# --- Сквозной сценарий по документам -----------------------------------------------------------


@activity.defn
async def prepare_context_activity(
    job_id: str, document_ids: list[str], query: str | None
) -> dict[str, Any]:
    """Поиск, защита и сборка контекста — один шаг."""
    _bind(job_id)
    context, report = await rag_pipeline.retrieve_context_async(query, document_ids)
    return {
        "context": context,
        "chunk_ids": sorted(context_chunk_ids(context)),
        "guard": report.to_dict(),
    }


@activity.defn
async def rag_generate_activity(
    job_id: str, context: str, query: str | None, feedback: list[str] | None
) -> dict[str, Any]:
    _bind(job_id)
    card = await rag_pipeline.generate_with_context(query, context, feedback)
    return card.model_dump()


@activity.defn
def verify_sources_activity(
    job_id: str, card: dict[str, Any], context: str, scores: dict[str, float]
) -> dict[str, Any]:
    _bind(job_id)
    draft, errors = rag_pipeline.check_sources(CardDraft.model_validate(card), context, scores)
    return {"ok": not errors, "errors": errors, "card": draft.model_dump()}


@activity.defn
async def rag_critique_activity(job_id: str, card: dict[str, Any], context: str) -> dict[str, Any]:
    _bind(job_id)
    report = await rag_pipeline.critique_with_context(context, CardDraft.model_validate(card))
    return report.model_dump()


@activity.defn
def finalize_rag_activity(
    job_id: str,
    card: dict[str, Any] | None,
    verdict: str,
    issues: list[str],
    source_errors: list[str],
    guard: dict[str, Any],
    attempts: int,
) -> dict[str, Any]:
    """Итог по документам: выдуманные ссылки после переделки и низкая уверенность — к человеку."""
    _bind(job_id)
    report = guard_from_dict(guard)
    draft = CardDraft.model_validate(card) if card else None
    reasons: list[str] = []
    if draft is None:
        status = "needs_review"
        reasons.append("карточка не собрана: в контексте нет пригодных фрагментов")
    elif source_errors:
        status = "needs_review"
        reasons.append("ссылки на источники не прошли проверку: " + "; ".join(source_errors[:3]))
        draft = rag_pipeline.apply_output_guard(draft, report)
    else:
        draft = rag_pipeline.apply_output_guard(draft, report)
        status = rag_pipeline.decide_status(draft, verdict, report)
        if status == "rejected":
            reasons.append("критик не одобрил за отведённые круги: " + "; ".join(issues[:3]))
        if report.needs_review:
            reasons.append(f"подозрительных фрагментов: {len(report.suspicious_chunks)}")
        if report.output_issues:
            reasons.append("фильтр выхода: " + "; ".join(report.output_issues[:3]))
        if status == "needs_review" and draft.confidence < 1 and not reasons:
            reasons.append(f"уверенность {draft.confidence:.2f} ниже порога")
    result: dict[str, Any] = draft.model_dump() if draft else {}
    result.update(
        security_report=report.to_dict(),
        attempts=attempts,
        verdict=verdict,
    )
    reason = "; ".join(reasons) or None
    with connection() as conn:
        jobs_repo.set_status(conn, job_id, status, result=result, review_reason=reason)
    log.info("job_finalized", status=status, reason=reason)
    return {"status": status, "review_reason": reason}


ALL_ACTIVITIES = [
    set_job_status,
    extract_activity,
    generate_activity,
    critique_activity,
    finalize_text_activity,
    parse_document_activity,
    index_document_activity,
    prepare_context_activity,
    rag_generate_activity,
    verify_sources_activity,
    rag_critique_activity,
    finalize_rag_activity,
]
