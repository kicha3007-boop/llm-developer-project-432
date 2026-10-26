"""Сервис документов: приём файла, разбор → нормализация → резка → сохранение → индексация.

Состояния документа: new → parsing → indexing → indexed, либо failed с внятной причиной
(например, «нет текстового слоя» у скана). Повторная загрузка того же содержимого не создаёт
дубля: идентификатор — префикс sha256 содержимого.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.parsers import SUPPORTED_EXTENSIONS, ParsedDocument, ParseError, parse_file
from app.parsers.chunker import Chunk, chunk_document
from app.parsers.normalizer import normalize
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
    ext = Path(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UploadRejected(
            f"формат «{ext or 'без расширения'}» не поддерживается; "
            f"принимаются: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if size == 0:
        raise UploadRejected("пустой файл")
    if size > settings.upload_max_bytes:
        raise UploadRejected(
            f"файл больше допустимого: {size} байт > {settings.upload_max_bytes} байт"
        )
    return ext


def register_upload(filename: str, content: bytes) -> UploadResult:
    """Сохраняет файл и регистрирует документ (идемпотентно по содержимому)."""
    ext = validate_upload(filename, len(content))
    sha = hashlib.sha256(content).hexdigest()
    document_id = sha[:12]
    storage = Path(settings.upload_dir)
    storage.mkdir(parents=True, exist_ok=True)
    path = storage / f"{document_id}{ext}"
    if not path.exists():
        path.write_bytes(content)
    document, created = documents_repo.get_or_create(
        document_id=document_id,
        sha256=sha,
        filename=Path(filename).name,
        extension=ext,
        size_bytes=len(content),
        storage_path=str(path),
    )
    log.info("document_registered", document_id=document["id"], created=created)
    return UploadResult(document=document, created=created)


def document_title(parsed: ParsedDocument, chunks: list[Chunk]) -> str | None:
    """Заголовок для поиска: у таблиц xlsx заголовка нет — берём лист и бренд из строк."""
    if parsed.fmt != "xlsx":
        return parsed.title
    brands = sorted({c.metadata["brand"] for c in chunks if c.metadata.get("brand")})
    return " ".join(x for x in [parsed.title, ", ".join(brands)] if x) or None


def parse_document(document_id: str) -> dict[str, Any]:
    """Шаг разбора: файл → фрагменты в базе. Блокирующий — в воркере идёт в потоке."""
    document = documents_repo.get(document_id)
    if document is None:
        raise LookupError(f"документ {document_id} не найден")
    if document["status"] == DocumentStatus.INDEXED:
        total, _ = chunks_repo.counts(document_id)
        return {"document_id": document_id, "status": document["status"], "chunks": total}
    documents_repo.set_status(document_id, DocumentStatus.PARSING)
    try:
        parsed = normalize(parse_file(document["storage_path"]))
        chunks = chunk_document(parsed, document_id, document["filename"])
        if not chunks:
            raise ParseError("в документе не найдено текста для индексации")
    except ParseError as exc:
        documents_repo.set_status(document_id, DocumentStatus.FAILED, error=exc.reason)
        log.warning("document_failed", document_id=document_id, reason=exc.reason)
        return {"document_id": document_id, "status": DocumentStatus.FAILED, "error": exc.reason}
    saved = chunks_repo.replace_for_document(document_id, chunks)
    documents_repo.set_status(
        document_id,
        DocumentStatus.INDEXING,
        title=document_title(parsed, chunks),
        pages=parsed.pages,
    )
    log.info("document_parsed", document_id=document_id, chunks=saved)
    return {"document_id": document_id, "status": DocumentStatus.INDEXING, "chunks": saved}


def index_document(document_id: str) -> dict[str, Any]:
    """Шаг индексации: досчитать векторы фрагментам документа. Повтор — пустая работа."""
    document = documents_repo.get(document_id)
    if document is None:
        raise LookupError(f"документ {document_id} не найден")
    if document["status"] == DocumentStatus.FAILED:
        return {
            "document_id": document_id,
            "status": document["status"],
            "error": document["error"],
        }
    embedded = reindex_missing(document_id=document_id)
    documents_repo.set_status(document_id, DocumentStatus.INDEXED)
    total, with_vec = chunks_repo.counts(document_id)
    log.info("document_indexed", document_id=document_id, embedded_now=embedded, total=total)
    return {
        "document_id": document_id,
        "status": DocumentStatus.INDEXED,
        "chunks": total,
        "embedded": with_vec,
    }


def process_document(document_id: str) -> dict[str, Any]:
    """Разбор и индексация подряд (для команды и тестов; в воркере — два отдельных шага)."""
    result = parse_document(document_id)
    if result["status"] == DocumentStatus.FAILED:
        return result
    return index_document(document_id)


def describe(document_id: str) -> dict[str, Any] | None:
    document = documents_repo.get(document_id)
    if document is None:
        return None
    total, embedded = chunks_repo.counts(document_id)
    return {
        "document_id": document["id"],
        "filename": document["filename"],
        "status": document["status"],
        "error": document["error"],
        "title": document["title"],
        "pages": document["pages"],
        "chunk_count": total,
        "embedded_count": embedded,
    }
