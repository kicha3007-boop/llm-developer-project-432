"""Устойчивые процессы на живом Temporal и живой базе (пропускаются без инфраструктуры).

Модель подменена; воркер поднимается в процессе теста на отдельной очереди.
"""

import asyncio
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from temporalio.client import Client
from temporalio.worker import Worker

from app.core.config import settings
from app.rag import embedder
from app.repositories import jobs as jobs_repo
from app.services import documents as documents_service
from app.temporal.activities import ALL_ACTIVITIES
from app.temporal.workflows import ALL_WORKFLOWS
from tests.conftest import DATA, as_json
from tests.fakes import HashEmbedder, delete_document, delete_job

pytestmark = [pytest.mark.db, pytest.mark.integration]

FACTS = as_json(
    {"product_name": "Блендер", "characteristics": {"Мощность": "800 Вт"}, "missing_fields": []}
)
CARD = as_json(
    {
        "title": "Блендер 800 Вт",
        "characteristics": {"Мощность": "800 Вт"},
        "missing_fields": [],
        "confidence": 0.9,
    }
)
APPROVE = as_json({"verdict": "approve", "issues": []})


@pytest.fixture
async def temporal_client(live_db):
    try:
        return await asyncio.wait_for(Client.connect(settings.temporal_host), timeout=3)
    except Exception:
        pytest.skip("Temporal недоступен (make infra) — интеграционный тест пропущен")


def _worker(client: Client, queue: str, executor: ThreadPoolExecutor) -> Worker:
    return Worker(
        client,
        task_queue=queue,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
        activity_executor=executor,
        graceful_shutdown_timeout=timedelta(seconds=1),
    )


async def _wait_status(job_id: str, statuses: set[str], timeout: float = 30) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        row = await asyncio.to_thread(jobs_repo.get_job, job_id)
        if row and row["status"] in statuses:
            return row
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"статус {row and row['status']} не дошёл до {statuses}")
        await asyncio.sleep(0.2)


async def test_card_workflow_waits_for_human_and_signal_finishes(temporal_client, fake_llm):
    fake_llm.role("extractor", FACTS).role("generator", CARD).role("critic", APPROVE)
    queue = f"test-{uuid.uuid4().hex[:6]}"
    job_id, _ = jobs_repo.create_job("card", {"supplier_text": "Блендер 800 Вт"})
    try:
        with ThreadPoolExecutor(4) as executor:
            async with _worker(temporal_client, queue, executor):
                handle = await temporal_client.start_workflow(
                    "CardWorkflow",
                    args=[job_id, "Блендер 800 Вт", 3],
                    id=f"job-{job_id}",
                    task_queue=queue,
                )
                row = await _wait_status(job_id, {"approved", "needs_review"})
                assert row["status"] == "approved"
                assert row["attempts"] == 1
                state = await handle.query("current_status")
                assert "awaiting human decision" in state["status"]
                await handle.signal("approve", "ок")
                assert await handle.result() == "confirmed"
        assert jobs_repo.get_job(job_id)["status"] == "confirmed"
    finally:
        delete_job(job_id)


async def test_worker_killed_midway_resumes_without_repeating_steps(
    temporal_client, fake_llm, monkeypatch
):
    """Воркер убит во время генерации: новый воркер доводит процесс, извлечение не повторяется."""
    generate_started = asyncio.Event()
    calls = {"generate": 0}

    fake_llm.role("extractor", FACTS).role("critic", APPROVE)

    original = fake_llm.__call__

    async def scripted(spec, prompt, *, schema=None, purpose=None):
        if spec.name == "generator":
            calls["generate"] += 1
            if calls["generate"] == 1:
                generate_started.set()
                await asyncio.sleep(3600)  # «зависший» вызов — воркер умрёт раньше
            return CARD
        return await original(spec, prompt, schema=schema, purpose=purpose)

    from app.llm import client as llm_client

    monkeypatch.setattr(llm_client, "run_agent", scripted)
    queue = f"test-{uuid.uuid4().hex[:6]}"
    job_id, _ = jobs_repo.create_job("card", {"supplier_text": "Блендер"})
    try:
        with ThreadPoolExecutor(4) as executor:
            worker = _worker(temporal_client, queue, executor)
            task = asyncio.create_task(worker.run())
            handle = await temporal_client.start_workflow(
                "CardWorkflow", args=[job_id, "Блендер", 3], id=f"job-{job_id}", task_queue=queue
            )
            await asyncio.wait_for(generate_started.wait(), timeout=30)
            await worker.shutdown()  # «убиваем» воркер посреди шага генерации
            await asyncio.gather(task, return_exceptions=True)

            async with _worker(temporal_client, queue, executor):
                row = await _wait_status(job_id, {"approved", "needs_review"}, timeout=60)
                await handle.signal("reject", "не нравится")
                assert await handle.result() == "rejected"
        assert row["status"] == "approved"
        assert fake_llm.roles().count("extractor") == 1  # выполненный шаг не повторялся
        assert calls["generate"] == 2
        history = await handle.fetch_history()
        scheduled = [
            e.activity_task_scheduled_event_attributes.activity_type.name
            for e in history.events
            if e.HasField("activity_task_scheduled_event_attributes")
        ]
        assert scheduled.count("extract_activity") == 1
    finally:
        delete_job(job_id)


def _cite(spec, prompt):
    blocks = re.findall(r"\[(\w+) \| [^\]]*\]\n([^\[]*)", prompt)
    power = next(cid for cid, text in blocks if "Мощность: 800 Вт" in text)
    return as_json(
        {
            "title": "Погружной блендер МиксерПро 800",
            "characteristics": {"Мощность": "800 Вт"},
            "sources": [{"chunk_id": power, "field": "Мощность"}],
            "missing_fields": [],
            "confidence": 0.9,
        }
    )


async def test_generate_card_workflow_end_to_end(temporal_client, fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(documents_service.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(settings, "relevance_threshold", 0.0)
    embedder.set_embedder(HashEmbedder())
    fake_llm.role("rag_generator", _cite).role("rag_critic", APPROVE)
    name = "blender_passport.pdf"
    doc_id = documents_service.register_upload(name, (DATA / name).read_bytes()).document["id"]
    job_id, _ = jobs_repo.create_job("rag", {"document_ids": [doc_id]})
    queue = f"test-{uuid.uuid4().hex[:6]}"
    try:
        with ThreadPoolExecutor(4) as executor:
            async with _worker(temporal_client, queue, executor):
                handle = await temporal_client.start_workflow(
                    "GenerateCardWorkflow",
                    args=[job_id, [doc_id], "блендер", 3],
                    id=f"job-{job_id}",
                    task_queue=queue,
                )
                row = await _wait_status(job_id, {"approved", "needs_review", "failed"}, 60)
                assert row["status"] == "approved", row["review_reason"] or row["error"]
                assert row["result"]["sources"][0]["page"] == 1
                assert "security_report" in row["result"]
                await handle.signal("approve")
                assert await handle.result() == "confirmed"
    finally:
        embedder.set_embedder(None)
        delete_job(job_id)
        delete_document(doc_id)
