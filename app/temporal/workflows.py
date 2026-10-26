"""Устойчивые процессы: генерация по тексту, разбор документа, сквозная генерация по документам.

Код процесса исполняется в песочнице и при восстановлении проигрывается с начала, поэтому:
- импорты сервисов пропускаются мимо песочницы (``imports_passed_through``);
- таймауты и политики повторов — константы модуля, а не настройки из окружения;
- состояние процесса — поля объекта, без обращений к базе: в базу пишут шаги.

Ожидание решения человека — ``wait_condition`` на сигнал: процесс в это время не занимает
ни потока, ни соединения и может ждать сутками.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from app.schemas.jobs import JobStatus
    from app.temporal.activities import (
        critique_activity,
        critique_verdict,
        extract_activity,
        finalize_rag_activity,
        finalize_text_activity,
        generate_activity,
        index_document_activity,
        parse_document_activity,
        prepare_context_activity,
        rag_critique_activity,
        rag_generate_activity,
        set_job_status,
        verify_sources_activity,
    )

DB_TIMEOUT = timedelta(seconds=30)
LLM_TIMEOUT = timedelta(minutes=10)
PARSE_TIMEOUT = timedelta(minutes=5)
INDEX_TIMEOUT = timedelta(minutes=30)  # первый запуск качает модель эмбеддингов

# Повтор ВЫЗОВА модели делает клиент. Здесь — повтор ШАГА: упал воркер, база, сеть до сервера.
# Детерминированные ошибки (невалидный ответ после лечения) не повторяются — это счёт впустую.
LLM_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=1),
    maximum_attempts=3,
    non_retryable_error_types=["CardValidationError", "ValidationError"],
)
DB_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=10,
)
PARSE_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    maximum_attempts=3,
    non_retryable_error_types=["LookupError"],
)


class _HumanReview:
    """Общая часть: ожидание решения человека, сигналы и запрос состояния."""

    def __init__(self) -> None:
        self._status = "starting"
        self._attempts = 0
        self._decision: str | None = None
        self._comment: str | None = None

    @workflow.signal
    def approve(self, comment: str | None = None) -> None:
        self._decision = "approve"
        self._comment = comment

    @workflow.signal
    def reject(self, comment: str | None = None) -> None:
        self._decision = "reject"
        self._comment = comment

    @workflow.query
    def current_status(self) -> dict[str, Any]:
        return {"status": self._status, "attempts": self._attempts, "decision": self._decision}

    async def _status_step(self, job_id: str, status: str, **kwargs: Any) -> None:
        self._status = status
        await workflow.execute_activity(
            set_job_status,
            args=[
                job_id,
                status,
                kwargs.get("result"),
                kwargs.get("error"),
                kwargs.get("review_reason"),
                kwargs.get("bump_attempts", False),
            ],
            start_to_close_timeout=DB_TIMEOUT,
            retry_policy=DB_RETRY,
        )

    async def _await_human(self, job_id: str, auto_status: str) -> str:
        self._status = f"{auto_status}: awaiting human decision"
        await workflow.wait_condition(lambda: self._decision is not None)
        final = JobStatus.CONFIRMED if self._decision == "approve" else JobStatus.REJECTED
        await self._status_step(job_id, final, review_reason=self._comment)
        return final

    async def _fail(self, job_id: str, exc: BaseException) -> str:
        cause = exc.cause if isinstance(exc, ActivityError) and exc.cause else exc
        await self._status_step(job_id, JobStatus.FAILED, error=str(cause)[:1000])
        return JobStatus.FAILED


@workflow.defn
class CardWorkflow(_HumanReview):
    """Пайплайн по тексту: извлечение → (генерация → проверка) × ≤N → решение → человек."""

    @workflow.run
    async def run(self, job_id: str, supplier_text: str, max_rounds: int = 3) -> str:
        try:
            await self._status_step(job_id, JobStatus.EXTRACTING)
            facts = await workflow.execute_activity(
                extract_activity,
                args=[job_id, supplier_text],
                start_to_close_timeout=LLM_TIMEOUT,
                retry_policy=LLM_RETRY,
            )
            feedback: list[str] = []
            verdict = "rejected"
            card: dict[str, Any] = {}
            while self._attempts < max_rounds:
                self._attempts += 1
                await self._status_step(
                    job_id,
                    JobStatus.GENERATING if self._attempts == 1 else JobStatus.REGENERATING,
                    bump_attempts=True,
                )
                card = await workflow.execute_activity(
                    generate_activity,
                    args=[job_id, facts, feedback or None],
                    start_to_close_timeout=LLM_TIMEOUT,
                    retry_policy=LLM_RETRY,
                )
                await self._status_step(job_id, JobStatus.VERIFYING)
                report = await workflow.execute_activity(
                    critique_activity,
                    args=[job_id, facts, card],
                    start_to_close_timeout=LLM_TIMEOUT,
                    retry_policy=LLM_RETRY,
                )
                approved, feedback = critique_verdict(report)
                if approved:
                    verdict = "approved"
                    feedback = []
                    break
            decision = await workflow.execute_activity(
                finalize_text_activity,
                args=[job_id, card, verdict, feedback, self._attempts],
                start_to_close_timeout=DB_TIMEOUT,
                retry_policy=DB_RETRY,
            )
        except ActivityError as exc:
            return await self._fail(job_id, exc)
        return await self._await_human(job_id, decision["status"])


@workflow.defn
class DocumentWorkflow:
    """Разбор и индексация загруженного документа: два шага, каждый в потоке воркера."""

    def __init__(self) -> None:
        self._status = "new"

    @workflow.query
    def current_status(self) -> str:
        return self._status

    @workflow.run
    async def run(self, document_id: str) -> dict[str, Any]:
        self._status = "parsing"
        parsed = await workflow.execute_activity(
            parse_document_activity,
            args=[document_id, None],
            start_to_close_timeout=PARSE_TIMEOUT,
            retry_policy=PARSE_RETRY,
        )
        if parsed["status"] == "failed":
            self._status = "failed"
            return parsed
        self._status = "indexing"
        indexed = await workflow.execute_activity(
            index_document_activity,
            args=[document_id, None],
            start_to_close_timeout=INDEX_TIMEOUT,
            retry_policy=PARSE_RETRY,
        )
        self._status = indexed["status"]
        return indexed


@workflow.defn
class GenerateCardWorkflow(_HumanReview):
    """Сквозной сценарий: разбор → индексация → поиск → генерация → проверка → человек."""

    @workflow.run
    async def run(
        self, job_id: str, document_ids: list[str], product_hint: str | None, max_rounds: int = 3
    ) -> str:
        try:
            await self._status_step(job_id, JobStatus.PARSING)
            for doc_id in document_ids:
                await workflow.execute_activity(
                    parse_document_activity,
                    args=[doc_id, job_id],
                    start_to_close_timeout=PARSE_TIMEOUT,
                    retry_policy=PARSE_RETRY,
                )
            await self._status_step(job_id, JobStatus.INDEXING)
            for doc_id in document_ids:
                await workflow.execute_activity(
                    index_document_activity,
                    args=[doc_id, job_id],
                    start_to_close_timeout=INDEX_TIMEOUT,
                    retry_policy=PARSE_RETRY,
                )
            await self._status_step(job_id, JobStatus.RETRIEVING)
            prepared = await workflow.execute_activity(
                prepare_context_activity,
                args=[job_id, document_ids, product_hint],
                start_to_close_timeout=LLM_TIMEOUT,
                retry_policy=LLM_RETRY,
            )
            context: str = prepared["context"]
            scores: dict[str, float] = prepared["guard"].get("retrieval_scores", {})
            feedback: list[str] = []
            verdict = "rejected"
            violations = 0
            source_errors: list[str] = []
            card: dict[str, Any] | None = None
            while prepared["chunk_ids"] and self._attempts < max_rounds:
                self._attempts += 1
                await self._status_step(
                    job_id,
                    JobStatus.GENERATING if self._attempts == 1 else JobStatus.REGENERATING,
                    bump_attempts=True,
                )
                card = await workflow.execute_activity(
                    rag_generate_activity,
                    args=[job_id, context, product_hint, feedback or None],
                    start_to_close_timeout=LLM_TIMEOUT,
                    retry_policy=LLM_RETRY,
                )
                await self._status_step(job_id, JobStatus.VERIFYING)
                check = await workflow.execute_activity(
                    verify_sources_activity,
                    args=[job_id, card, context, scores],
                    start_to_close_timeout=DB_TIMEOUT,
                    retry_policy=DB_RETRY,
                )
                card = check["card"]
                if not check["ok"]:
                    violations += 1
                    source_errors = feedback = check["errors"]
                    if violations >= 2:  # выдуманная ссылка повторно — к человеку
                        break
                    continue
                violations = 0
                source_errors = []
                report = await workflow.execute_activity(
                    rag_critique_activity,
                    args=[job_id, card, context],
                    start_to_close_timeout=LLM_TIMEOUT,
                    retry_policy=LLM_RETRY,
                )
                approved, feedback = critique_verdict(report)
                if approved:
                    verdict = "approved"
                    feedback = []
                    break
            decision = await workflow.execute_activity(
                finalize_rag_activity,
                args=[
                    job_id,
                    card,
                    verdict,
                    feedback,
                    source_errors,
                    prepared["guard"],
                    self._attempts,
                ],
                start_to_close_timeout=DB_TIMEOUT,
                retry_policy=DB_RETRY,
            )
        except ActivityError as exc:
            return await self._fail(job_id, exc)
        return await self._await_human(job_id, decision["status"])


ALL_WORKFLOWS = [CardWorkflow, DocumentWorkflow, GenerateCardWorkflow]
