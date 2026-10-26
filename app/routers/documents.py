"""Загрузка документов поставщиков, их состояние и поиск по фрагментам."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.rag import retrieval
from app.schemas.jobs import DocumentOut, DocumentStatus, SearchHit, SearchRequest
from app.services import documents as documents_service
from app.temporal import client as temporal

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger(__name__)

UPLOAD_FILE = File(...)


@router.post("/", status_code=202)
async def upload_document(file: UploadFile = UPLOAD_FILE) -> JSONResponse:
    content = await file.read(settings.upload_max_bytes + 1)
    try:
        result = await run_in_threadpool(
            documents_service.register_upload, file.filename or "", content
        )
    except documents_service.UploadRejected as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    document = result.document
    if document["status"] == DocumentStatus.NEW:
        try:
            await temporal.start_document_workflow(document["id"])
        except Exception as exc:  # документ разберёт и процесс генерации, но предупредим
            log.warning(
                "document_workflow_not_started", document_id=document["id"], error=repr(exc)
            )
    return JSONResponse(
        {"document_id": document["id"], "status": document["status"]}, status_code=202
    )


@router.post("/search", response_model=list[SearchHit])
def search(body: SearchRequest) -> list[SearchHit]:
    hits = retrieval.search(
        body.query,
        mode=body.mode,
        top_k=body.top_k,
        document_ids=body.document_ids,
        articul=body.articul,
        brand=body.brand,
        threshold=body.threshold,
    )
    return [
        SearchHit(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            text=h.text,
            score=h.score,
            page=h.page,
            section=h.section,
            metadata=h.metadata,
        )
        for h in hits
    ]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str) -> DocumentOut:
    info = documents_service.describe(document_id)
    if info is None:
        raise HTTPException(404, "документ не найден")
    return DocumentOut(**info)
