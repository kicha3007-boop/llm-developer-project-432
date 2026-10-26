"""Шаги процессов Temporal — тонкие обёртки над сервисами.

Единица восстановления в устойчивом исполнении — шаг: выполненный шаг записан в историю и после
рестарта воркера заново не выполняется. Поэтому стадии — отдельные шаги, а не один большой.

Шаги, которые не отдают управление (база, диск, разбор документа, эмбеддинги), объявлены
синхронными и исполняются в пуле потоков воркера. Шаги с вызовом модели — асинхронные.
Каждый шаг привязывает ``job_id`` к контексту логов: поиск по нему показывает весь путь задачи.
"""

from __future__ import annotations

import asyncio
from typing import Any

from temporalio import activity

from app.core.logging import bind_context, get_logger
from app.rag.context import BuiltContext, ContextItem, SecurityReport, build_context
from app.repositories import jobs as jobs_repo
from app.schemas.cards import CritiqueReport, ProductCard, SupplierFacts
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
    jobs_repo.set_status(
        job_id,
        status,
        result=result,
        error=error,
        review_reason=review_reason,
        bump_attempts=bump_attempts,
    )
    log.info("job_status", status=status)


# --- Пайплайн по тексту (шаг «Архитектура») ----------------------------------------------------


@activity.defn
async def extract_activity(job_id: str, supplier_text: str) -> dict[str, Any]:
    _bind(job_id)
    facts = await pipeline.extract(supplier_text)
    return facts.model_dump()


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
        SupplierFacts.model_validate(facts), ProductCard.model_validate(card)
    )
    return report.model_dump()


# --- Документы ---------------------------------------------------------------------------------


@activity.defn
def parse_document_activity(document_id: str, job_id: str | None = None) -> dict[str, Any]:
    _bind(job_id, document_id=document_id)
    return documents_service.parse_document(document_id)


@activity.defn
def index_document_activity(document_id: str, job_id: str | None = None) -> dict[str, Any]:
    _bind(job_id, document_id=document_id)
    return documents_service.index_document(document_id)


# --- Сквозной сценарий по документам -----------------------------------------------------------


def context_to_dict(ctx: BuiltContext) -> dict[str, Any]:
    return {
        "text": ctx.text,
        "items": [vars(i) for i in ctx.items],
        "security": ctx.security.to_dict(),
        "duplicates_dropped": ctx.duplicates_dropped,
        "truncated": ctx.truncated,
    }


def context_from_dict(data: dict[str, Any]) -> BuiltContext:
    return BuiltContext(
        text=data["text"],
        items=[ContextItem(**i) for i in data["items"]],
        security=SecurityReport(**data["security"]),
        duplicates_dropped=data.get("duplicates_dropped", 0),
        truncated=data.get("truncated", 0),
    )


@activity.defn
async def prepare_context_activity(
    job_id: str, document_ids: list[str], product_hint: str | None
) -> dict[str, Any]:
    """Поиск и сборка контекста (с маскированием и детектором инъекций) — один шаг."""
    _bind(job_id)
    hits = await asyncio.to_thread(rag_pipeline.retrieve, document_ids, product_hint)
    ctx = await build_context(hits)
    return context_to_dict(ctx)


@activity.defn
async def rag_generate_activity(
    job_id: str, context: dict[str, Any], product_hint: str | None, feedback: list[str] | None
) -> dict[str, Any]:
    _bind(job_id)
    card = await rag_pipeline.generate_from_context(
        context_from_dict(context), product_hint, feedback
    )
    return card.model_dump()


@activity.defn
def verify_sources_activity(
    job_id: str, card: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    _bind(job_id)
    check = rag_pipeline.check_sources(
        ProductCard.model_validate(card), context_from_dict(context).items
    )
    return {
        "ok": check.ok,
        "errors": check.errors,
        "card": check.card.model_dump() if check.card else card,
    }


@activity.defn
async def rag_critique_activity(
    job_id: str, card: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    _bind(job_id)
    report = await rag_pipeline.critique_against_sources(
        context_from_dict(context), ProductCard.model_validate(card)
    )
    return report.model_dump()


@activity.defn
def finalize_text_activity(
    job_id: str, card: dict[str, Any], verdict: str, issues: list[str], attempts: int
) -> dict[str, Any]:
    """Решение по пайплайну из текста: согласовано или к человеку; запись результата."""
    _bind(job_id)
    product = ProductCard.model_validate(card)
    status, reason = pipeline.decide(product, verdict, issues)
    result = {**product.model_dump(), "attempts": attempts}
    if issues:
        result["issues"] = issues
    jobs_repo.set_status(job_id, status, result=result, review_reason=reason)
    log.info("job_finalized", status=status, reason=reason)
    return {"status": status, "review_reason": reason}


@activity.defn
def finalize_rag_activity(
    job_id: str,
    card: dict[str, Any] | None,
    verdict: str,
    issues: list[str],
    source_failures: int,
    context: dict[str, Any],
    attempts: int,
) -> dict[str, Any]:
    _bind(job_id)
    ctx = context_from_dict(context)
    product = ProductCard.model_validate(card) if card else None
    status, reason, product = rag_pipeline.decide(
        product,
        verdict=verdict,
        issues=issues,
        source_failures=source_failures,
        security=ctx.security,
    )
    result = rag_pipeline.result_payload(product, ctx.security, ctx.chunk_ids, attempts, issues)
    jobs_repo.set_status(job_id, status, result=result, review_reason=reason)
    log.info("job_finalized", status=status, reason=reason)
    return {"status": status, "review_reason": reason}


def critique_verdict(report: dict[str, Any]) -> tuple[bool, list[str]]:
    parsed = CritiqueReport.model_validate(report)
    return parsed.verdict == "approve", pipeline.issue_texts(parsed)


ALL_ACTIVITIES = [
    set_job_status,
    extract_activity,
    generate_activity,
    critique_activity,
    parse_document_activity,
    index_document_activity,
    prepare_context_activity,
    rag_generate_activity,
    verify_sources_activity,
    rag_critique_activity,
    finalize_text_activity,
    finalize_rag_activity,
]
