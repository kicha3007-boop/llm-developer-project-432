"""HTTP-слой: коды ответов, идемпотентность, отказ на неподдерживаемом файле.

TestClient без контекстного менеджера не запускает жизненный цикл: пул базы здесь не нужен,
репозитории подменены."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.repositories import documents as documents_repo
from app.repositories import jobs as jobs_repo
from app.temporal import client as temporal
from tests.conftest import as_json


@pytest.fixture
def memory_jobs(monkeypatch):
    """Таблица задач в памяти вместо базы."""
    rows: dict[str, dict] = {}
    keys: dict[str, str] = {}

    def create_job(kind, payload, key=None):
        if key and key in keys:
            return keys[key], False
        job_id = f"job{len(rows) + 1}"
        rows[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "pending",
            "payload": payload,
            "result": None,
            "attempts": 0,
            "error": None,
            "review_reason": None,
            "created_at": None,
            "updated_at": None,
        }
        if key:
            keys[key] = job_id
        return job_id, True

    def set_status(job_id, status, **kw):
        rows[job_id]["status"] = status
        rows[job_id].update({k: v for k, v in kw.items() if k in rows[job_id] and v is not None})

    monkeypatch.setattr(jobs_repo, "create_job", create_job)
    monkeypatch.setattr(jobs_repo, "get_job", lambda job_id: rows.get(job_id))
    monkeypatch.setattr(jobs_repo, "set_status", set_status)
    monkeypatch.setattr(jobs_repo, "set_workflow_id", lambda job_id, wf: None)
    monkeypatch.setattr(
        documents_repo, "get_many", lambda ids: [{"id": i} for i in ids if i.startswith("doc")]
    )
    return rows


@pytest.fixture
def started(monkeypatch):
    calls = []

    async def fake_start(job_id, document_ids, hint):
        calls.append((job_id, document_ids, hint))
        return f"job-{job_id}"

    async def fake_card(job_id, text):
        calls.append((job_id, text))
        return f"job-{job_id}"

    monkeypatch.setattr(temporal, "start_generate_workflow", fake_start)
    monkeypatch.setattr(temporal, "start_card_workflow", fake_card)
    return calls


def test_generate_card_202_and_idempotency(memory_jobs, started):
    body = {"document_ids": ["doc1", "doc2"], "product_hint": "блендер"}
    client = TestClient(app)
    first = client.post("/generate-card", json=body, headers={"Idempotency-Key": "k1"})
    second = client.post("/generate-card", json=body, headers={"Idempotency-Key": "k1"})
    status = client.get(f"/jobs/{first.json()['job_id']}")
    assert first.status_code == 202 and second.status_code == 202
    assert first.json() == {"job_id": "job1", "status": "pending"}
    assert second.json()["job_id"] == "job1"  # вторая задача не создана
    assert len(started) == 1  # и второй процесс не запущен
    assert status.json()["status"] == "pending"


def test_generate_card_unknown_document(memory_jobs, started):
    client = TestClient(app)
    response = client.post("/generate-card", json={"document_ids": ["nope"]})
    assert response.status_code == 404
    assert started == []


def test_engine_unavailable_marks_job_failed(memory_jobs, monkeypatch):
    async def broken(*args):
        raise ConnectionError("temporal down")

    monkeypatch.setattr(temporal, "start_card_workflow", broken)
    client = TestClient(app)
    response = client.post("/jobs", json={"supplier_text": "Блендер"})
    assert response.status_code == 503
    assert memory_jobs["job1"]["status"] == "failed"


def test_get_unknown_job_404(memory_jobs):
    client = TestClient(app)
    assert client.get("/jobs/zzz").status_code == 404


def test_upload_rejects_unsupported_format():
    client = TestClient(app)
    response = client.post("/documents/", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert "не поддерживается" in response.json()["detail"]


def test_sync_card_endpoint(fake_llm):
    fake_llm.role(
        "extractor",
        as_json(
            {
                "product_name": "Чайник",
                "characteristics": {"Мощность": "2200 Вт"},
                "missing_fields": ["Цвет"],
            }
        ),
    )
    fake_llm.role(
        "generator",
        as_json(
            {
                "title": "Чайник 2200 Вт",
                "characteristics": {"Мощность": "2200 Вт"},
                "missing_fields": ["Цвет"],
                "confidence": 0.9,
            }
        ),
    )
    fake_llm.role("critic", as_json({"verdict": "approve", "issues": []}))
    client = TestClient(app)
    response = client.post("/cards", json={"supplier_text": "Чайник 2200 Вт"})
    assert response.status_code == 200
    data = response.json()
    assert data["card"]["missing_fields"] == ["Цвет"]
    assert data["card"]["confidence"] < 1.0
    assert data["status"] in {"approved", "needs_review"}


def test_sync_card_invalid_model_answer_is_not_500(fake_llm):
    fake_llm.add(*["не json"] * 10)
    client = TestClient(app)
    response = client.post("/cards", json={"supplier_text": "текст"})
    assert response.status_code == 422
