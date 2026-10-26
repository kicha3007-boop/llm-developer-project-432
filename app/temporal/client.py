"""Клиент Temporal для приложения: запуск процессов по имени, сигналы, запрос состояния.

Приложение не импортирует код процессов: запускает их по имени строкой. Идентификатор процесса
задаём сами (``job-<id задачи>``): повторный запуск с тем же ключом не создаст второй процесс.
"""

from __future__ import annotations

import asyncio
from typing import Any

from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_client: Client | None = None
_lock = asyncio.Lock()


class WorkflowNotFound(LookupError):
    pass


async def get_client() -> Client:
    global _client
    if _client is None:
        async with _lock:
            if _client is None:
                _client = await Client.connect(
                    settings.temporal_host, namespace=settings.temporal_namespace
                )
    return _client


def job_workflow_id(job_id: str) -> str:
    return f"job-{job_id}"


def document_workflow_id(document_id: str) -> str:
    return f"doc-{document_id}"


async def _start(workflow: str, args: list[Any], workflow_id: str) -> str:
    client = await get_client()
    try:
        await client.start_workflow(
            workflow, args=args, id=workflow_id, task_queue=settings.temporal_task_queue
        )
        log.info("workflow_started", workflow=workflow, workflow_id=workflow_id)
    except WorkflowAlreadyStartedError:
        log.info("workflow_already_started", workflow=workflow, workflow_id=workflow_id)
    return workflow_id


async def start_card_workflow(job_id: str, supplier_text: str) -> str:
    return await _start(
        "CardWorkflow",
        [job_id, supplier_text, settings.max_pipeline_rounds],
        job_workflow_id(job_id),
    )


async def start_generate_workflow(
    job_id: str, document_ids: list[str], product_hint: str | None
) -> str:
    return await _start(
        "GenerateCardWorkflow",
        [job_id, document_ids, product_hint, settings.max_pipeline_rounds],
        job_workflow_id(job_id),
    )


async def start_document_workflow(document_id: str) -> str:
    return await _start("DocumentWorkflow", [document_id], document_workflow_id(document_id))


async def signal(job_id: str, name: str, comment: str | None = None) -> None:
    client = await get_client()
    handle = client.get_workflow_handle(job_workflow_id(job_id))
    try:
        await handle.signal(name, comment)
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise WorkflowNotFound(job_id) from exc
        raise


async def describe(job_id: str) -> dict[str, Any]:
    """Состояние выполнения от сервера плюс внутренний статус процесса через запрос."""
    client = await get_client()
    handle = client.get_workflow_handle(job_workflow_id(job_id))
    try:
        info = await handle.describe()
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise WorkflowNotFound(job_id) from exc
        raise
    state: Any = None
    if info.status == WorkflowExecutionStatus.RUNNING:
        state = await handle.query("current_status")
    return {
        "workflow_id": handle.id,
        "execution_status": info.status.name if info.status else None,
        "state": state,
    }
