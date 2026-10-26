"""Каркас: миграции, пул, проверки состояния, репозиторий задач."""

from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.core import db
from app.core.config import settings
from app.core.migrate import MIGRATIONS_DIR, discover, migrate, pending
from app.main import app
from app.repositories import jobs as jobs_repo
from tests.fakes import delete_job


def test_migrations_discovered_in_order():
    names = [m.name for m in discover()]
    assert names[0] == "0001_init"
    assert names == sorted(names)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in discover()[0].sql()
    assert "schema_migrations" in discover()[0].sql()


def test_pending_skips_applied():
    all_m = discover()
    assert [m.version for m in pending(all_m, {"0001", "0002"})] == [m.version for m in all_m[2:]]


def test_bad_migration_name(tmp_path: Path):
    (tmp_path / "init.sql").write_text("SELECT 1")
    with pytest.raises(ValueError, match="NNNN_name"):
        discover(tmp_path)


@pytest.mark.db
def test_migrate_from_scratch_then_noop(live_db, unique):
    """На чистой базе применяются все миграции, повторный запуск ничего не применяет."""
    dbname = f"card_test_{unique}"
    admin = psycopg.connect(settings.database_url, autocommit=True)
    admin.execute(f'CREATE DATABASE "{dbname}"')
    try:
        url = settings.database_url.rsplit("/", 1)[0] + f"/{dbname}"
        first = migrate(url)
        assert first == [m.name for m in discover(MIGRATIONS_DIR)]
        assert migrate(url) == []
        with psycopg.connect(url) as conn:
            ext = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            assert ext is not None
    finally:
        admin.execute(f'DROP DATABASE "{dbname}" WITH (FORCE)')
        admin.close()


@pytest.mark.db
def test_pool_is_shared_resource(live_db):
    assert db.pool() is db.pool()
    with db.connection() as conn:
        assert conn.execute("SELECT 1 AS one").fetchone()["one"] == 1


# --- проверки состояния ------------------------------------------------------------------------


def test_liveness_ok_even_if_db_down(monkeypatch):
    def broken():
        raise psycopg.OperationalError("db down")

    monkeypatch.setattr(db, "readiness", broken)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not ready"


def test_readiness_reports_vector(monkeypatch):
    monkeypatch.setattr(db, "readiness", lambda: {"db": "ok", "vector": "0.8.1"})
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "db": "ok", "vector": "0.8.1"}


def test_request_id_header():
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "req-42"})
    assert response.headers["X-Request-ID"] == "req-42"


@pytest.mark.db
def test_readiness_on_live_db(live_db):
    state = db.readiness()
    assert state["db"] == "ok" and state["vector"]


# --- репозиторий задач -------------------------------------------------------------------------


@pytest.mark.db
def test_jobs_idempotency_and_status(live_db, unique):
    key = f"key-{unique}"
    job_id, created = jobs_repo.create_job("card", {"supplier_text": "x"}, key)
    try:
        again, created_again = jobs_repo.create_job("card", {"supplier_text": "x"}, key)
        assert created and not created_again and again == job_id
        jobs_repo.set_status(job_id, "generating", bump_attempts=True)
        jobs_repo.set_status(job_id, "generating", bump_attempts=True)
        jobs_repo.set_status(job_id, "needs_review", result={"title": "Т"}, review_reason="низко")
        row = jobs_repo.get_job(job_id)
        assert row["status"] == "needs_review"
        assert row["attempts"] == 2
        assert row["result"] == {"title": "Т"}  # JSONB приходит словарём
        assert row["review_reason"] == "низко"
    finally:
        delete_job(job_id)


@pytest.mark.db
def test_jobs_without_key_are_distinct(live_db):
    a, _ = jobs_repo.create_job("card", {}, None)
    b, _ = jobs_repo.create_job("card", {}, None)
    try:
        assert a != b
    finally:
        delete_job(a)
        delete_job(b)
