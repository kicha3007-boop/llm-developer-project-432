"""Сервис документов на живой базе: загрузка выданных файлов, отказ скана, идемпотентность."""

import pytest

from app.rag import embedder
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.services import documents as documents_service
from app.services.documents import UploadRejected, validate_upload
from tests.conftest import DATA
from tests.fakes import HashEmbedder, delete_document

FILES = ["blender_passport.pdf", "blender_kp.docx", "kettle_spec.xlsx", "kettle_manual.pdf"]


def test_upload_validation():
    assert validate_upload("a.PDF", 10) == "pdf"
    with pytest.raises(UploadRejected, match="не поддерживается"):
        validate_upload("a.doc", 10)
    with pytest.raises(UploadRejected, match="пустой"):
        validate_upload("a.pdf", 0)
    with pytest.raises(UploadRejected, match="больше допустимого"):
        validate_upload("a.pdf", 10**10)


@pytest.fixture
def env(live_db, tmp_path, monkeypatch):
    monkeypatch.setattr(documents_service.settings, "upload_dir", str(tmp_path))
    embedder.set_embedder(HashEmbedder())
    created: list[str] = []
    yield created
    embedder.set_embedder(None)
    for doc_id in created:
        delete_document(doc_id)


def upload(env, name: str) -> str:
    doc_id = documents_service.register_upload(name, (DATA / name).read_bytes()).document["id"]
    env.append(doc_id)
    return doc_id


@pytest.mark.db
@pytest.mark.parametrize("name", FILES)
def test_given_files_get_indexed(env, conn, name):
    doc_id = upload(env, name)
    assert documents_service.ingest_document(doc_id) == "indexed"
    stored = documents_repo.get_document(conn, doc_id)
    assert stored["status"] == "indexed" and stored["kind"] == name.rsplit(".", 1)[1]
    assert stored["chunks_count"] > 0 and stored["embedded_count"] == stored["chunks_count"]
    for chunk in chunks_repo.list_chunks(conn, doc_id):
        assert chunk["metadata"]["doc_id"] == doc_id
        assert "page" in chunk["metadata"] and chunk["section"]
    # повторная загрузка того же файла — тот же документ, без дублей фрагментов
    again = documents_service.register_upload(name, (DATA / name).read_bytes())
    assert again.document["id"] == doc_id and not again.created
    documents_service.ingest_document(doc_id)
    assert chunks_repo.count_chunks(conn, doc_id) == stored["chunks_count"]


@pytest.mark.db
def test_xlsx_articuls_and_title(env, conn):
    doc_id = upload(env, "kettle_spec.xlsx")
    documents_service.ingest_document(doc_id)
    chunks = chunks_repo.list_chunks(conn, doc_id)
    assert [c["metadata"].get("articul") for c in chunks] == ["KTL-1700", "KTL-1000", "KTL-THRM"]
    assert documents_repo.get_document(conn, doc_id)["title"] == "Спецификация КеттлПро"


@pytest.mark.db
def test_scan_fails_with_reason(env, conn):
    doc_id = upload(env, "boiler_scan.pdf")
    assert documents_service.ingest_document(doc_id) == "failed"
    stored = documents_repo.get_document(conn, doc_id)
    assert "нет текстового слоя" in stored["error"]
    assert chunks_repo.count_chunks(conn, doc_id) == 0


@pytest.mark.db
def test_reindex_second_run_is_noop(env):
    from app.rag.reindex import reindex_missing

    doc_id = upload(env, "kettle_spec.xlsx")
    documents_service.parse_document(doc_id)
    assert reindex_missing(document_id=doc_id) == 3
    assert reindex_missing(document_id=doc_id) == 0
