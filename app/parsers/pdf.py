"""Разбор pdf: текст постранично строками с кеглем и положением, таблицы — отдельными блоками.

Таблицы извлекаются по разметке страницы (pdfplumber) вместе с заголовками колонок; строки текста,
попавшие внутрь таблицы, в текстовый поток не дублируются. Документ без текстового слоя
(скан) получает отказ с причиной, а не пустой результат.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

import pdfplumber

from app.parsers import Block, NoTextLayerError, ParsedDocument, ParseError


def _inside(top: float, bottom: float, bbox: tuple[float, float, float, float]) -> bool:
    center = (top + bottom) / 2
    return bbox[1] - 1 <= center <= bbox[3] + 1


def _clean_cell(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _pick_title(lines: list[Block]) -> str | None:
    """Заголовок документа — самая крупная строка первой страницы; надпись капсом вроде
    «ПАСПОРТ ИЗДЕЛИЯ» пропускается, если есть другая крупная строка."""
    ranked = sorted(lines, key=lambda b: (-(b.size or 0), b.top or 0))[:3]
    for line in ranked:
        if not line.text.isupper():
            return line.text
    return ranked[0].text if ranked else None


def parse_pdf(path: str | Path) -> ParsedDocument:
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:  # повреждённый файл
        raise ParseError(f"pdf не открывается: {exc}") from exc

    blocks: list[Block] = []
    total_chars = 0
    image_pages = 0
    title: str | None = None
    with pdf:
        pages = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            total_chars += len(page.chars)
            if page.images and not page.chars:
                image_pages += 1
            page_items: list[Block] = []
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]
            for table in tables:
                rows = [[_clean_cell(c) for c in row] for row in table.extract()]
                rows = [r for r in rows if any(r)]
                if rows:
                    page_items.append(
                        Block(kind="table", text="", page=page_no, rows=rows, top=table.bbox[1])
                    )
            for line in page.extract_text_lines(return_chars=True):
                if any(_inside(line["top"], line["bottom"], b) for b in bboxes):
                    continue
                text = line["text"].strip()
                if not text:
                    continue
                sizes = [c["size"] for c in line["chars"] if c.get("size")]
                page_items.append(
                    Block(
                        kind="line",
                        text=text,
                        page=page_no,
                        size=round(median(sizes), 1) if sizes else None,
                        top=line["top"],
                    )
                )
            page_items.sort(key=lambda b: b.top or 0.0)
            if page_no == 1:
                title = _pick_title([b for b in page_items if b.kind == "line" and b.size])
            blocks.extend(page_items)

    if total_chars == 0:
        raise NoTextLayerError(
            f"нет текстового слоя: документ похож на скан ({pages} стр., "
            f"{image_pages} из них — только изображение). Распознавание (OCR) не включено, "
            "загрузите pdf с текстовым слоем."
        )
    return ParsedDocument(blocks=blocks, pages=pages, title=title, fmt="pdf")
