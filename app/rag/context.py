"""Сборка контекста генерации из найденных фрагментов.

Порядок важен:
1. повторы убираются (одинаковые фрагменты из разных документов не съедают контекст);
2. персональные данные маскируются — до любого обращения к модели (в том числе к детектору
   инъекций) и до записи контекста в лог;
3. детектор инъекций выбрасывает подозрительные фрагменты; если в документе их больше порога,
   документ целиком уходит к человеку с причиной;
4. оставшиеся фрагменты оформляются как опознаваемые источники
   «[chunk_id | файл | стр. N | секция]» в пределах лимита символов.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.guardrails import injection
from app.guardrails.pii import PiiMasker
from app.repositories.chunks import Hit

log = get_logger(__name__)

Detector = Callable[[str, dict[str, Any] | None], Awaitable[injection.Detection]]


@dataclass
class ContextItem:
    chunk_id: str
    document_id: str
    filename: str | None
    page: int | None
    section: str | None
    text: str


@dataclass
class SecurityReport:
    excluded: list[dict[str, Any]] = field(default_factory=list)
    masked: dict[str, int] = field(default_factory=dict)
    flagged_documents: list[dict[str, Any]] = field(default_factory=list)
    output_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuiltContext:
    text: str
    items: list[ContextItem]
    security: SecurityReport
    duplicates_dropped: int = 0
    truncated: int = 0

    @property
    def chunk_ids(self) -> list[str]:
        return [i.chunk_id for i in self.items]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe(hits: Sequence[Hit]) -> tuple[list[Hit], int]:
    seen: set[str] = set()
    ids: set[str] = set()
    result = []
    for hit in hits:
        key = _norm(hit.text)
        if key in seen or hit.chunk_id in ids:
            continue
        seen.add(key)
        ids.add(hit.chunk_id)
        result.append(hit)
    return result, len(hits) - len(result)


def format_item(item: ContextItem) -> str:
    page = f"стр. {item.page}" if item.page is not None else "стр. —"
    head = " | ".join([item.chunk_id, item.filename or item.document_id, page, item.section or "—"])
    return f"[{head}]\n{item.text}"


def format_context(items: Sequence[ContextItem], max_chars: int) -> tuple[str, list[ContextItem]]:
    """Оформляет фрагменты до предела размера; фрагмент целиком или никак."""
    parts: list[str] = []
    included: list[ContextItem] = []
    size = 0
    for item in items:
        block = format_item(item)
        extra = len(block) + (2 if parts else 0)
        if size + extra > max_chars:
            continue
        parts.append(block)
        included.append(item)
        size += extra
    return "\n\n".join(parts), included


async def build_context(
    hits: Sequence[Hit],
    *,
    max_chars: int | None = None,
    masker: PiiMasker | None = None,
    detector: Detector | None = None,
    max_suspicious_per_doc: int | None = None,
) -> BuiltContext:
    max_chars = max_chars or settings.context_max_chars
    masker = masker or PiiMasker()
    detector = detector or injection.detect
    limit = (
        settings.injection_max_suspicious_per_doc
        if max_suspicious_per_doc is None
        else max_suspicious_per_doc
    )
    report = SecurityReport()

    unique, dropped = dedupe(hits)

    masked_items: list[tuple[Hit, str]] = [(h, masker.mask(h.text)) for h in unique]
    report.masked = masker.summary()

    kept: list[tuple[Hit, str]] = []
    suspicious_by_doc: dict[str, int] = {}
    for hit, text in masked_items:
        detection = await detector(text, hit.metadata)
        if detection.suspicious:
            suspicious_by_doc[hit.document_id] = suspicious_by_doc.get(hit.document_id, 0) + 1
            report.excluded.append(
                {
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "reasons": detection.reasons,
                    "level": detection.level,
                }
            )
            log.warning("context_chunk_excluded", chunk_id=hit.chunk_id, reasons=detection.reasons)
            continue
        kept.append((hit, text))

    for doc_id, count in suspicious_by_doc.items():
        if count > limit:
            report.flagged_documents.append(
                {
                    "document_id": doc_id,
                    "suspicious_chunks": count,
                    "reason": f"подозрительных фрагментов {count} — больше порога {limit}",
                }
            )
    flagged = {d["document_id"] for d in report.flagged_documents}
    kept = [(h, t) for h, t in kept if h.document_id not in flagged]

    items = [
        ContextItem(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            filename=h.metadata.get("filename"),
            page=h.page,
            section=h.section,
            text=t,
        )
        for h, t in kept
    ]
    text, included = format_context(items, max_chars)
    log.info(
        "context_built",
        chunks=len(included),
        duplicates_dropped=dropped,
        truncated=len(items) - len(included),
        excluded=len(report.excluded),
        masked=report.masked,
        chars=len(text),
    )
    return BuiltContext(
        text=text,
        items=included,
        security=report,
        duplicates_dropped=dropped,
        truncated=len(items) - len(included),
    )
