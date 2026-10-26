"""Разбор docx: параграфы и таблицы в порядке документа.

Заголовок (стили Heading N / Title) — граница секции: по нему потом видно, из какой части
документа взят фрагмент. Страниц у docx нет, номер считается по явным разрывам страниц.
"""

from __future__ import annotations

import io
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.parsers import Block, ParseError, read_source


def _heading_level(paragraph: Paragraph) -> int | None:
    style = (paragraph.style.name if paragraph.style is not None else "") or ""
    if style == "Title":
        return 0
    if style.startswith("Heading"):
        tail = style.removeprefix("Heading").strip()
        return int(tail) if tail.isdigit() else 1
    return None


def _page_breaks(element) -> int:
    breaks = sum(1 for br in element.iter(qn("w:br")) if br.get(qn("w:type")) == "page")
    return breaks + len(list(element.iter(qn("w:lastRenderedPageBreak"))))


def parse_docx(source: bytes | str | Path) -> list[Block]:
    try:
        document = docx.Document(io.BytesIO(read_source(source)))
    except Exception as exc:
        raise ParseError(f"docx не открывается: {exc}") from exc

    blocks: list[Block] = []
    page = 1
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
                blocks.append(
                    Block(kind="heading", content=text, metadata={"page": page, "level": level})
                )
            else:
                blocks.append(Block(kind="text", content=text, metadata={"page": page}))
        elif tag == "tbl":
            table = Table(element, document)
            rows = []
            for row in table.rows:
                cells = [" ".join(c.text.split()) for c in row.cells]
                if any(cells):
                    rows.append(cells)
            if rows:
                blocks.append(
                    Block(
                        kind="table",
                        content="\n".join(" | ".join(r) for r in rows),
                        rows=rows,
                        metadata={"page": page},
                    )
                )
    return blocks
