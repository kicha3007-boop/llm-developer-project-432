"""Сервис документов: приём файла, разбор → нормализация → резка → сохранение → индексация.

Состояния документа: new → parsing → indexing → indexed, либо failed с внятной причиной
(например, «нет текстового слоя» у скана). Повторная загрузка того же содержимого не создаёт
дубля: идентификатор — префикс sha256 содержимого. Соединение с базой берётся на каждую короткую
операцию, а не на всё время разбора.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.db import connection
from app.core.logging import get_logger
from app.parsers import SUPPORTED_KINDS, Block, ParseError, parse_any
from app.parsers.chunker import chunk_blocks
from app.parsers.normalizer import guess_title, normalize
from app.rag.reindex import reindex_missing
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.schemas.jobs import DocumentStatus

log = get_logger(__name__)


class UploadRejected(ValueError):
    """Файл не принят: неподдерживаемое расширение, пустой или слишком большой."""


@dataclass
class UploadResult:
    document: dict[str, Any]
    created: bool


def validate_upload(filename: str, size: int) -> str:
    """Проверяет расширение и размер; возвращает формат (pdf | docx | xlsx)."""
    kind = Path(filename or "").suffix.lower().lstrip(".")
    if kind not in SUPPORTED_KINDS:
        raise UploadRejected(
            f"формат «{kind or 'без расширения'}» не поддерживается; "
            f"принимаются: {', '.join(SUPPORTED_KINDS)}"
        )
    if size == 0:
        raise UploadRejected("пустой файл")
    if size > settings.upload_max_bytes:
        raise UploadRejected(
            f"файл больше допустимого: {size} байт > {settings.upload_max_bytes} байт"
        )
    return kind


def register_upload(filename: str, content: bytes) -> UploadResult:
    """Сохраняет файл и регистрирует документ (идемпотентно по содержимому)."""
    kind = validate_upload(filename, len(content))
    content_hash = hashlib.sha256(content).hexdigest()
    document_id = content_hash[:12]
    storage = Path(settings.upload_dir)
    storage.mkdir(parents=True, exist_ok=True)
    path = storage / f"{document_id}.{kind}"
    if not path.exists():
        path.write_bytes(content)
    with connection() as conn:
        document, created = documents_repo.create_document(
            conn,
            document_id=document_id,
            filename=Path(filename).name,
            content_hash=content_hash,
            kind=kind,
            size_bytes=len(content),
            stored_path=str(path),
        )
    log.info("document_registered", document_id=document["id"], created=created)
    return UploadResult(document=document, created=created)


def document_title(blocks: list[Block], chunks: list[dict[str, Any]]) -> str | None:
    """Заголовок для поиска; у таблиц xlsx заголовка нет — лист и бренд из строк."""
    title = guess_title(blocks)
    if any(b.metadata.get("sheet") for b in blocks):
        brands = sorted({c["metadata"]["brand"] for c in chunks if c["metadata"].get("brand")})
        title = " ".join(x for x in [title, ", ".join(brands)] if x) or None
    return title


def parse_to_chunks(
    data: bytes, kind: str, doc_id: str, filename: str | None = None
) -> list[dict[str, Any]]:
    """Байты файла → нормализованные фрагменты с метаданными (без базы)."""
    return chunk_blocks(normalize(parse_any(data, kind)), doc_id, filename)


def parse_document(document_id: str) -> dict[str, Any]:
    """Шаг разбора: файл → фрагменты в базе. Блокирующий — в воркере идёт в потоке."""
    with connection() as conn:
        document = documents_repo.get_document(conn, document_id)
    if document is None:
        raise LookupError(f"документ {document_id} не найден")
    if document["status"] == DocumentStatus.INDEXED:
        return {
            "document_id": document_id,
            "status": document["status"],
            "chunks": document["chunks_count"],
        }
    with connection() as conn:
        documents_repo.set_status(conn, document_id, DocumentStatus.PARSING)
    try:
        if not document["stored_path"]:
            raise ParseError("файл документа не сохранён")
        blocks = normalize(parse_any(Path(document["stored_path"]), document["kind"]))
        chunks = chunk_blocks(blocks, document_id, document["filename"])
        if not chunks:
            raise ParseError("в документе не найдено текста для индексации")
    except ParseError as exc:
        with connection() as conn:
            documents_repo.set_status(conn, document_id, DocumentStatus.FAILED, error=exc.reason)
        log.warning("document_failed", document_id=document_id, reason=exc.reason)
        return {"document_id": document_id, "status": DocumentStatus.FAILED, "error": exc.reason}
    pages = max((b.page or 0 for b in blocks), default=0) or None
    with connection() as conn:
        saved = chunks_repo.replace_chunks(conn, document_id, chunks)
        documents_repo.set_status(
            conn,
            document_id,
            DocumentStatus.INDEXING,
            title=document_title(blocks, chunks),
            pages=pages if document["kind"] != "xlsx" else None,
        )
    log.info("document_parsed", document_id=document_id, chunks=saved)
    return {"document_id": document_id, "status": DocumentStatus.INDEXING, "chunks": saved}


def index_document(document_id: str) -> int:
    """Шаг индексации: досчитать векторы фрагментам документа. Повтор — пустая работа."""
    return reindex_missing(document_id=document_id)


def mark_indexed(document_id: str) -> str:
    with connection() as conn:
        document = documents_repo.get_document(conn, document_id)
        if document is None or document["status"] == DocumentStatus.FAILED:
            return DocumentStatus.FAILED
        documents_repo.set_status(conn, document_id, DocumentStatus.INDEXED)
    log.info("document_indexed", document_id=document_id)
    return DocumentStatus.INDEXED


def ingest_document(document_id: str) -> str:
    """Разбор и индексация подряд; возвращает итоговое состояние (indexed | failed)."""
    if parse_document(document_id)["status"] == DocumentStatus.FAILED:
        return DocumentStatus.FAILED
    index_document(document_id)
    return mark_indexed(document_id)


def describe(document_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        document = documents_repo.get_document(conn, document_id)
    if document is None:
        return None
    return {
        "document_id": document["id"],
        "filename": document["filename"],
        "status": document["status"],
        "error": document["error"],
        "title": document["title"],
        "pages": document["pages"],
        "chunk_count": document["chunks_count"],
        "embedded_count": document["embedded_count"],
    }
