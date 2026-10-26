"""Резка нормализованного документа на фрагменты с метаданными.

- Текст режется фиксированным размером с перекрытием (утверждение на границе не теряется),
  в пределах одной страницы и секции.
- Таблица не рвёт строку. Спецификация «артикул → параметры» даёт фрагмент на строку вида
  «BLD-800: Мощность: 800 Вт; Объём: 1,5 л»; таблица «параметр — значение» режется частями,
  и каждая часть несёт заголовки колонок.
- Метаданные каждого фрагмента: документ, файл, страница, секция, вид, артикул, бренд.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.parsers import Block, ParsedDocument

ARTICUL_RE = re.compile(r"\b[A-ZА-Я]{2,6}-[A-Z0-9]{2,8}\b")
_UNIT_RE = re.compile(r"^(.+?),\s*([^\s,]{1,4})$")  # «Мощность, Вт» → («Мощность», «Вт»)
_ARTICUL_HEADERS = {"артикул", "арт.", "sku", "код товара", "артикул товара"}
_BRAND_HEADERS = {"бренд", "марка", "торговая марка", "производитель"}


@dataclass
class Chunk:
    id: str
    ord: int
    kind: str  # text | table
    text: str
    page: int | None
    section: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_id(document_id: str, ord_: int, text: str) -> str:
    """Детерминированный идентификатор: повторный разбор того же файла даёт те же id."""
    return hashlib.sha1(f"{document_id}:{ord_}:{text}".encode()).hexdigest()[:12]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Режет по словам на куски не длиннее ``size`` с перекрытием ``overlap`` символов."""
    if len(text) <= size:
        return [text]
    words = text.split()
    pieces: list[str] = []
    start = 0
    while start < len(words):
        length = 0
        end = start
        while end < len(words) and length + len(words[end]) + (1 if end > start else 0) <= size:
            length += len(words[end]) + (1 if end > start else 0)
            end += 1
        if end == start:  # слово длиннее size
            end = start + 1
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        # откатываемся назад на overlap символов, но обязательно продвигаемся вперёд
        back, acc = end, 0
        while back > start + 1 and acc + len(words[back - 1]) + 1 <= overlap:
            acc += len(words[back - 1]) + 1
            back -= 1
        start = back if back > start else end
    return pieces


def _header_unit(header: str) -> tuple[str, str | None]:
    match = _UNIT_RE.match(header)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return header.strip(), None


def format_cell(header: str, value: str) -> str:
    name, unit = _header_unit(header)
    if unit:
        return f"{name}: {value} {unit}"
    return f"{name}: {value}"


def _find_col(headers: list[str], names: set[str]) -> int | None:
    for i, h in enumerate(headers):
        if h.strip().lower() in names:
            return i
    return None


def _table_chunks(block: Block) -> list[tuple[str, dict[str, Any]]]:
    rows = block.rows or []
    if not rows:
        return []
    headers, body = rows[0], rows[1:]
    section = block.section or "Таблица"
    art_col = _find_col(headers, _ARTICUL_HEADERS)
    brand_col = _find_col(headers, _BRAND_HEADERS)
    result: list[tuple[str, dict[str, Any]]] = []

    if art_col is not None:
        # спецификация: фрагмент на строку, строка самодостаточна благодаря заголовкам
        for row_no, row in enumerate(body, start=2):
            articul = row[art_col]
            parts = [
                format_cell(headers[i], v)
                for i, v in enumerate(row)
                if i != art_col and v and i < len(headers)
            ]
            text = f"{articul}: " + "; ".join(parts)
            meta: dict[str, Any] = {"articul": articul, "row": row_no, "table": "spec"}
            if brand_col is not None and row[brand_col]:
                meta["brand"] = row[brand_col]
            result.append((text, meta))
        return result

    head_line = f"Таблица «{section}» (колонки: {' | '.join(headers)})"
    if len(headers) == 2:
        lines = [f"{k}: {v}" for k, v in body if k or v]
        kv = {k.strip().lower(): v for k, v in body}
        table_meta: dict[str, Any] = {"table": "params"}
        articul = next((kv[h] for h in _ARTICUL_HEADERS if h in kv), None)
        if articul:
            table_meta["articul"] = articul
        brand = next((kv[h] for h in _BRAND_HEADERS if h in kv), None)
        if brand:
            table_meta["brand"] = brand
        elif kv.get("модель"):
            table_meta["brand"] = kv["модель"].split()[0]
    else:
        lines = [
            "; ".join(format_cell(h, v) for h, v in zip(headers, row, strict=False) if v)
            for row in body
        ]
        table_meta = {"table": "generic"}

    # части таблицы не длиннее chunk_size, строка не рвётся, заголовки — в каждой части
    part: list[str] = []
    size = len(head_line)
    for line in lines:
        if part and size + len(line) + 1 > settings.chunk_size:
            result.append(("\n".join([head_line, *part]), dict(table_meta)))
            part, size = [], len(head_line)
        part.append(line)
        size += len(line) + 1
    if part:
        result.append(("\n".join([head_line, *part]), dict(table_meta)))
    return result


def chunk_document(
    doc: ParsedDocument,
    document_id: str,
    filename: str,
    *,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    raw: list[tuple[str, str, int | None, str | None, dict[str, Any]]] = []

    # текстовые блоки собираем по (страница, секция), таблицы — отдельно, в порядке документа
    buffer: list[Block] = []

    def flush() -> None:
        if not buffer:
            return
        text = "\n".join(b.text for b in buffer)
        meta: dict[str, Any] = {}
        if any(b.meta.get("tiny_font") for b in buffer):
            meta["tiny_font"] = True
        for piece in split_text(text, size, overlap):
            raw.append(("text", piece, buffer[0].page, buffer[0].section, dict(meta)))
        buffer.clear()

    for block in doc.blocks:
        if block.kind == "heading":
            flush()
            continue
        if block.kind == "table":
            flush()
            for text, meta in _table_chunks(block):
                raw.append(("table", text, block.page, block.section, {**block.meta, **meta}))
            continue
        # граница фрагмента: другая страница, секция или резкая смена кегля (мелкий шрифт
        # уходит отдельным фрагментом — так подозрительная вставка не тянет за собой соседей)
        if buffer and (
            (buffer[-1].page, buffer[-1].section, bool(buffer[-1].meta.get("tiny_font")))
            != (block.page, block.section, bool(block.meta.get("tiny_font")))
        ):
            flush()
        buffer.append(block)
    flush()

    articuls = {m["articul"] for *_x, m in raw if m.get("articul")}
    doc_articul = next(iter(articuls)) if len(articuls) == 1 else None
    brands = {m["brand"] for *_x, m in raw if m.get("brand")}
    doc_brand = next(iter(brands)) if len(brands) == 1 else None

    chunks: list[Chunk] = []
    for ord_, (kind, text, page, section, meta) in enumerate(raw):
        metadata: dict[str, Any] = {
            "document_id": document_id,
            "filename": filename,
            "page": page,
            "section": section,
            "kind": kind,
            **meta,
        }
        in_text = sorted(set(ARTICUL_RE.findall(text)))
        if in_text:
            metadata["articuls"] = in_text
        if "articul" not in metadata:
            if doc_articul:
                metadata["articul"] = doc_articul
            elif len(in_text) == 1:
                metadata["articul"] = in_text[0]
        if "brand" not in metadata and doc_brand:
            metadata["brand"] = doc_brand
        chunks.append(
            Chunk(
                id=chunk_id(document_id, ord_, text),
                ord=ord_,
                kind=kind,
                text=text,
                page=page,
                section=section,
                metadata=metadata,
            )
        )
    return chunks
