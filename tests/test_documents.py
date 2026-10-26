"""Сервис документов на живой базе: загрузка выданных файлов, отказ скана, идемпотентность."""

import pytest

from app.core.db import connection
from app.rag import embedder
from app.services import documents as documents_service
from app.services.documents import UploadRejected, validate_upload
from tests.conftest import DATA
from tests.fakes import HashEmbedder, delete_document

FILES = ["blender_passport.pdf", "blender_kp.docx", "kettle_spec.xlsx", "kettle_manual.pdf"]


def test_upload_validation():
    assert validate_upload("a.PDF", 10) == ".pdf"
    with pytest.raises(UploadRejected, match="не поддерживается"):
        validate_upload("a.doc", 10)
    with pytest.raises(UploadRejected, match="пустой"):
        validate_upload("a.pdf", 0)
    with pytest.raises(UploadRejected, match="больше допустимого"):
        validate_upload("a.pdf", 10**10)


@pytest.fixture
def hash_embedder():
    embedder.set_embedder(HashEmbedder())
    yield
    embedder.set_embedder(None)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(documents_service.settings, "upload_dir", str(tmp_path))
    return tmp_path


@pytest.mark.db
@pytest.mark.parametrize("name", FILES)
def test_given_files_get_indexed(live_db, hash_embedder, storage, name):
    upload = documents_service.register_upload(name, (DATA / name).read_bytes())
    doc_id = upload.document["id"]
    try:
        result = documents_service.process_document(doc_id)
        assert result["status"] == "indexed"
        info = documents_service.describe(doc_id)
        assert info["chunk_count"] > 0
        assert info["embedded_count"] == info["chunk_count"]
        with connection() as conn:
            rows = conn.execute(
                "SELECT metadata, section FROM chunks WHERE document_id = %s", (doc_id,)
            ).fetchall()
        for row in rows:
            assert row["metadata"]["document_id"] == doc_id
            assert "page" in row["metadata"]
            assert row["section"]
        # повторная загрузка того же файла — тот же документ, без дублей фрагментов
        again = documents_service.register_upload(name, (DATA / name).read_bytes())
        assert again.document["id"] == doc_id and not again.created
        documents_service.process_document(doc_id)
        assert documents_service.describe(doc_id)["chunk_count"] == info["chunk_count"]
    finally:
        delete_document(doc_id)


@pytest.mark.db
def test_scan_fails_with_reason(live_db, hash_embedder, storage):
    name = "boiler_scan.pdf"
    upload = documents_service.register_upload(name, (DATA / name).read_bytes())
    doc_id = upload.document["id"]
    try:
        result = documents_service.process_document(doc_id)
        assert result["status"] == "failed"
        info = documents_service.describe(doc_id)
        assert info["status"] == "failed"
        assert "нет текстового слоя" in info["error"]
        assert info["chunk_count"] == 0
    finally:
        delete_document(doc_id)


@pytest.mark.db
def test_reindex_second_run_is_noop(live_db, hash_embedder, storage):
    from app.rag.reindex import reindex_missing

    name = "kettle_spec.xlsx"
    upload = documents_service.register_upload(name, (DATA / name).read_bytes())
    doc_id = upload.document["id"]
    try:
        documents_service.parse_document(doc_id)
        assert reindex_missing(document_id=doc_id) == 3
        assert reindex_missing(document_id=doc_id) == 0
    finally:
        delete_document(doc_id)
