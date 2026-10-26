"""Разбор docx: параграфы и таблицы в порядке документа.

Заголовок (стили Heading N / Title) — граница секции: по нему потом видно, из какой части документа
взят фрагмент. Страниц у docx нет, поэтому номер считается по явным разрывам страниц.
"""

from __future__ import annotations

from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.parsers import Block, ParsedDocument, ParseError


def _heading_level(paragraph: Paragraph) -> int | None:
    style = (paragraph.style.name if paragraph.style is not None else "") or ""
    if style == "Title":
        return 0
    if style.startswith("Heading"):
        tail = style.removeprefix("Heading").strip()
        return int(tail) if tail.isdigit() else 1
    return None


def _page_breaks(element) -> int:
    breaks = 0
    for br in element.iter(qn("w:br")):
        if br.get(qn("w:type")) == "page":
            breaks += 1
    breaks += len(list(element.iter(qn("w:lastRenderedPageBreak"))))
    return breaks


def parse_docx(path: str | Path) -> ParsedDocument:
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ParseError(f"docx не открывается: {exc}") from exc

    blocks: list[Block] = []
    page = 1
    title: str | None = None
    for element in document.element.body.iterchildren():
        tag = element.tag.split("}")[-1]
        if tag == "p":
            paragraph = Paragraph(element, document)
            page += _page_breaks(element)
            text = paragraph.text.strip()
            if not text:
                continue
            level = _heading_level(paragraph)
            if level is not None:
                blocks.append(Block(kind="heading", text=text, page=page, level=level))
                if title is None and level >= 2:
                    title = text
            else:
                blocks.append(Block(kind="text", text=text, page=page))
        elif tag == "tbl":
            table = Table(element, document)
            rows = []
            for row in table.rows:
                cells = [" ".join(c.text.split()) for c in row.cells]
                if any(cells):
                    rows.append(cells)
            if rows:
                blocks.append(Block(kind="table", text="", page=page, rows=rows))
    if title is None:
        title = next((b.text for b in blocks if b.kind == "heading"), None)
    return ParsedDocument(blocks=blocks, pages=page, title=title, fmt="docx")
