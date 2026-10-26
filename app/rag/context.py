"""Сборка контекста генерации из найденных (и уже проверенных защитой) фрагментов.

Каждый фрагмент опознаваем: начинается заголовком ``[chunk_id | файл | стр. N | секция]``.
Модель ссылается на ``chunk_id``, а приложение достаёт допустимые идентификаторы обратно из
текста контекста (``context_chunk_ids``) и сверяет с ними ссылки карточки. У контекста есть предел
размера, фрагменты с одинаковым текстом (из разных документов) не повторяются.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import settings

_HEADER_RE = re.compile(r"^\[([^\s|\]]+)[^\]\n]*\]\s*$", re.MULTILINE)
_PII_LABEL_RE = re.compile(r"^(PHONE|EMAIL|INN|CARD)_\d+$")  # метка маскирования — не источник


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def format_chunk(chunk: dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    filename = chunk.get("filename") or meta.get("filename") or meta.get("doc_id")
    page = chunk.get("page", meta.get("page"))
    section = chunk.get("section") or meta.get("section")
    parts = [str(chunk_id)]
    if filename:
        parts.append(str(filename))
    if page is not None:
        parts.append(f"стр. {page}")
    if section:
        parts.append(str(section))
    return f"[{' | '.join(parts)}]\n{chunk.get('text', '').strip()}"


def build_context(chunks: list[dict[str, Any]], max_chars: int | None = None) -> str:
    """Оформляет фрагменты как источники в пределах ``max_chars``; фрагмент целиком или никак."""
    limit = max_chars or settings.context_max_chars
    seen: set[str] = set()
    parts: list[str] = []
    size = 0
    for chunk in chunks:
        key = _norm(chunk.get("text", ""))
        if not key or key in seen:
            continue
        block = format_chunk(chunk)
        extra = len(block) + (2 if parts else 0)
        if size + extra > limit:
            continue
        seen.add(key)
        parts.append(block)
        size += extra
    return "\n\n".join(parts)


def context_chunk_ids(context: str) -> set[str]:
    """Идентификаторы фрагментов, которые действительно были в поданном контексте."""
    return {cid for cid in _HEADER_RE.findall(context) if not _PII_LABEL_RE.match(cid)}
