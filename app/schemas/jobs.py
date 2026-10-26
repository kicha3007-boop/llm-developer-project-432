"""Контракты задач и документов: состояния и формы ответов HTTP."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JobStatus:
    """Цепочка состояний задачи генерации.

    pending → parsing → indexing → retrieving → generating → verifying
            ↘ (переделка: regenerating → generating …, не больше max_pipeline_rounds кругов)
    → done          — критик одобрил, уверенность не ниже порога
    → needs_review  — ждёт человека: низкая уверенность, выдуманные ссылки, инъекции и т.п.
    → rejected      — критик не одобрил за отведённые круги (черновик сохранён)
    дальше процесс ждёт решения человека:
    → confirmed     — человек подтвердил карточку (сигнал approve)
    → rejected      — человек отклонил карточку (сигнал reject)
    → failed        — авария (исключение), отличается от делового отказа
    """

    PENDING = "pending"
    PARSING = "parsing"
    INDEXING = "indexing"
    RETRIEVING = "retrieving"
    EXTRACTING = "extracting"
    GENERATING = "generating"
    VERIFYING = "verifying"
    REGENERATING = "regenerating"
    DONE = "done"
    NEEDS_REVIEW = "needs_review"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    FAILED = "failed"

    # Автоматическая часть закончена, процесс ждёт решения человека
    AWAITING_HUMAN = frozenset({DONE, NEEDS_REVIEW, REJECTED})
    TERMINAL = frozenset({CONFIRMED, REJECTED, FAILED})


class DocumentStatus:
    NEW = "new"
    PARSING = "parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobOut(BaseModel):
    job_id: str
    kind: str
    status: str
    attempts: int
    result: dict[str, Any] | None = None
    error: str | None = None
    review_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentCreated(BaseModel):
    document_id: str
    status: str


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    status: str
    error: str | None = None
    title: str | None = None
    pages: int | None = None
    chunk_count: int = 0
    embedded_count: int = 0


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="hybrid", pattern="^(vector|text|hybrid)$")
    top_k: int = Field(default=8, ge=1, le=50)
    document_ids: list[str] | None = None
    articul: str | None = None
    brand: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionIn(BaseModel):
    comment: str | None = None
