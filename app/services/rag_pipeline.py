"""Сквозной сценарий: документы → поиск → защита → контекст → генерация → проверка → карточка.

Три рубежа против выдумывания:
1. поиск — модель видит только найденные фрагменты (``retrieve_context_async``);
2. проверка ссылок кодом (``verify_sources``) — источник обязан (а) существовать в базе и
   (б) входить в поданный контекст. Второе важнее: ссылка на существующий фрагмент, которого
   модель не видела, — выдумывание с правдоподобным видом. Нарушение — переделка с текстом
   ошибки; повторное — ``SourceValidationError`` (вызывающий уводит задачу к человеку);
3. критик сверяет утверждения карточки с заявленными источниками (``critique_with_context``).

``doc_id``, ``page``, ``section`` и ``score`` источника подставляет система из базы и поиска.
``run_rag_pipeline`` возвращает ``(черновик, попытки, вердикт, статус, отчёт защиты)``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from app.agents import prompts
from app.agents.prompts import CRITIC_CONTRACT, critic_agent, generator_agent
from app.core.config import settings
from app.core.db import connection
from app.core.logging import get_logger
from app.guardrails import GuardReport, guard_context, guard_output
from app.llm.client import run_agent
from app.rag.context import build_context, context_chunk_ids
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.schemas.cards import CardDraft, CritiqueReport
from app.schemas.jobs import JobStatus
from app.services import structured
from app.services.pipeline import cap_confidence, norm

__all__ = [
    "CRITIC_CONTRACT",
    "SourceValidationError",
    "critique_with_context",
    "decide_status",
    "generate_with_context",
    "retrieve_context_async",
    "run_rag_pipeline",
    "verify_sources",
]

log = get_logger(__name__)

# Запросы, которыми собираются сведения для карточки, кроме самого запроса и заголовков документов
FIELD_QUERIES = (
    "технические характеристики параметры",
    "мощность объём вес размеры материал",
    "гарантия страна-производитель цвет",
    "описание преимущества комплект поставки",
)

StatusCallback = Callable[[str], None]


class SourceValidationError(Exception):
    """Карточка ссылается на фрагменты, которых нет в базе или не было в контексте."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


# --- Поиск и контекст --------------------------------------------------------------------------


def retrieval_queries(query: str | None, titles: Sequence[str]) -> list[str]:
    queries: list[str] = [q for q in [query, *titles] if q]
    base = query or (titles[0] if titles else "")
    queries.extend(f"{base} {q}".strip() for q in FIELD_QUERIES)
    seen: set[str] = set()
    return [q for q in queries if not (q in seen or seen.add(q))]


def retrieve_hits(query: str | None, doc_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    """Гибридный поиск по запросу, заголовкам документов и типовым полям карточки, с фильтром
    по выбранным документам. Повторы фрагментов схлопываются, лучшая оценка сохраняется."""
    with connection() as conn:
        titles: list[str] = []
        if doc_ids:
            titles = [d["title"] for d in documents_repo.get_documents(conn, doc_ids) if d["title"]]
        found: dict[str, dict[str, Any]] = {}
        for q in retrieval_queries(query, titles):
            for hit in chunks_repo.search_hybrid(
                conn, q, settings.search_top_k, None, list(doc_ids) if doc_ids else None
            ):
                best = found.get(hit["chunk_id"])
                if best is None or hit["score"] > best["score"]:
                    found[hit["chunk_id"]] = hit
        hits = sorted(found.values(), key=lambda h: -h["score"])
        # Документы выбраны явно: остальные их фрагменты идут следом за найденными (в порядке
        # документа), чтобы строки спецификации, не прошедшие порог по общему запросу, не
        # терялись. Предел размера контекста соблюдает build_context.
        for doc_id in doc_ids or []:
            for row in chunks_repo.list_chunks(conn, doc_id):
                if row["id"] not in found and not row["suspicious"]:
                    meta = row["metadata"] or {}
                    hits.append(
                        {
                            "chunk_id": row["id"],
                            "doc_id": doc_id,
                            "text": row["text"],
                            "metadata": meta,
                            "page": row["page"] if row["page"] is not None else meta.get("page"),
                            "section": row["section"] or meta.get("section"),
                            "score": 0.0,
                        }
                    )
                    found[row["id"]] = hits[-1]
    log.info("retrieval_done", documents=len(doc_ids or []), hits=len(hits))
    return hits


async def retrieve_context_async(
    query: str | None, doc_ids: Sequence[str] | None = None
) -> tuple[str, GuardReport]:
    """Поиск → защита (маскирование и инъекции) → контекст с опознаваемыми фрагментами."""
    hits = await asyncio.to_thread(retrieve_hits, query, doc_ids)
    safe, report = await guard_context(hits)
    report.retrieval_scores = {h["chunk_id"]: round(float(h["score"]), 4) for h in hits}
    if report.suspicious_chunks:
        await asyncio.to_thread(_mark_suspicious, report.suspicious_chunks)
    context = build_context(safe)
    return context, report


def _mark_suspicious(chunk_ids: list[str]) -> None:
    with connection() as conn:
        documents_repo.mark_chunks_suspicious(conn, chunk_ids)


# --- Генерация и проверка ----------------------------------------------------------------------


async def generate_with_context(
    query: str | None, context: str, feedback: list[str] | None = None
) -> CardDraft:
    log.info("step_generate_start", feedback=bool(feedback))
    prompt = prompts.build_rag_generate_prompt(
        query or "товар из документов", context, sorted(context_chunk_ids(context)), feedback
    )
    card = await structured.validate_or_retry(prompt, agent=generator_agent, runner=run_agent)
    return cap_confidence(card)


def verify_sources(
    draft: CardDraft,
    context: str,
    existing: dict[str, dict[str, Any]],
    scores: dict[str, float] | None = None,
) -> tuple[CardDraft, list[str]]:
    """Ссылки проверяются по двум условиям: фрагмент есть в базе и был в поданном контексте.

    Возвращает карточку с обогащёнными источниками (doc_id, page, section, score из базы и
    поиска) и список нарушений.
    """
    in_context = context_chunk_ids(context)
    scores = scores or {}
    errors: list[str] = []
    enriched = []
    for src in draft.sources:
        if src.chunk_id not in existing:
            errors.append(f"источник {src.chunk_id} не существует в базе — ссылка выдумана")
            continue
        if src.chunk_id not in in_context:
            errors.append(
                f"источник {src.chunk_id} не входил в поданный контекст — модель его не видела"
            )
            continue
        row = existing[src.chunk_id]
        enriched.append(
            src.model_copy(
                update={
                    "doc_id": row.get("doc_id") or src.doc_id,
                    "page": row.get("page") if row.get("page") is not None else src.page,
                    "section": row.get("section") or src.section,
                    "score": scores.get(src.chunk_id, src.score),
                }
            )
        )
    if draft.characteristics and not draft.sources:
        errors.append("в карточке есть характеристики, но нет ни одного источника")
    elif draft.sources and all(s.field for s in draft.sources):
        covered = {norm(s.field or "") for s in draft.sources}
        errors.extend(
            f"у характеристики «{name}» нет источника в sources"
            for name in draft.characteristics
            if norm(name) not in covered
        )
    return draft.model_copy(update={"sources": enriched}), errors


def check_sources(
    draft: CardDraft, context: str, scores: dict[str, float] | None = None
) -> tuple[CardDraft, list[str]]:
    with connection() as conn:
        existing = chunks_repo.get_chunks_by_ids(conn, [s.chunk_id for s in draft.sources])
    return verify_sources(draft, context, existing, scores)


async def critique_with_context(context: str, draft: CardDraft) -> CritiqueReport:
    """Критик сверяет утверждения с фрагментами; контракт источников — в начале промпта."""
    log.info("step_critique_start")
    return await structured.ask_model(
        critic_agent,
        prompts.build_rag_critique_prompt(context, draft),
        CritiqueReport,
        runner=run_agent,
    )


def decide_status(draft: CardDraft | None, verdict: str, report: GuardReport) -> str:
    """Итоговое состояние после автоматических проверок."""
    if draft is None or verdict != "approved":
        return JobStatus.REJECTED
    if report.needs_review or report.output_issues:
        return JobStatus.NEEDS_REVIEW
    return structured.confidence_gate(draft)


def apply_output_guard(draft: CardDraft, report: GuardReport) -> CardDraft:
    """Фильтр выхода: контакты в карточке маскируются, замечания — в отчёт защиты."""
    cleaned, issues = guard_output(draft.model_dump_json())
    if not issues:
        return draft
    report.output_issues = issues
    return CardDraft.model_validate_json(cleaned)


async def run_rag_pipeline(
    query: str | None,
    doc_ids: Sequence[str] | None = None,
    *,
    max_rounds: int | None = None,
    on_status: StatusCallback | None = None,
) -> tuple[CardDraft, int, str, str, GuardReport]:
    """Весь сценарий в одном процессе (команды, метрики, тесты). В Temporal те же стадии
    идут отдельными шагами."""
    max_rounds = max_rounds or settings.max_pipeline_rounds

    def status(value: str) -> None:
        if on_status:
            on_status(value)

    status(JobStatus.RETRIEVING)
    context, report = await retrieve_context_async(query, doc_ids)
    if not context_chunk_ids(context):
        raise SourceValidationError(["в контексте нет пригодных фрагментов"])

    feedback: list[str] = []
    verdict = "rejected"
    violations = 0
    attempts = 0
    draft: CardDraft | None = None
    while attempts < max_rounds:
        attempts += 1
        status(JobStatus.GENERATING if attempts == 1 else JobStatus.REGENERATING)
        draft = await generate_with_context(query, context, feedback or None)
        status(JobStatus.VERIFYING)
        draft, errors = await asyncio.to_thread(
            check_sources, draft, context, report.retrieval_scores
        )
        if errors:
            violations += 1
            log.warning("sources_rejected", attempt=attempts, errors=errors)
            if violations >= 2:  # повторное нарушение — дальше решает вызывающий
                raise SourceValidationError(errors)
            feedback = errors
            continue
        violations = 0
        critique = await critique_with_context(context, draft)
        if critique.verdict == "approve":
            verdict = "approved"
            break
        feedback = critique.issues
        log.info("pipeline_regenerate", attempt=attempts, issues=feedback)
    if draft is None or violations:
        raise SourceValidationError(feedback or ["карточка не собрана"])
    draft = apply_output_guard(draft, report)
    final = decide_status(draft, verdict, report)
    log.info("rag_pipeline_done", status=final, attempts=attempts, verdict=verdict)
    return draft, attempts, verdict, final, report
