"""Разбор pdf: текст постранично строками (с кеглем и положением), таблицы — отдельными блоками.

Таблицы извлекаются по разметке страницы (pdfplumber) вместе с заголовками колонок; строки текста,
попавшие внутрь таблицы, в текстовый поток не дублируются. Документ без текстового слоя (скан)
получает ``PdfNoTextLayer`` с причиной, а не пустой результат.
"""

from __future__ import annotations

import io
from pathlib import Path
from statistics import median

import pdfplumber

from app.parsers import Block, ParseError, read_source


class PdfNoTextLayer(ParseError):
    """В pdf нет текстового слоя — похоже на скан. Распознавание (OCR) не включено."""


def _inside(top: float, bottom: float, bbox: tuple[float, float, float, float]) -> bool:
    center = (top + bottom) / 2
    return bbox[1] - 1 <= center <= bbox[3] + 1


def _cell(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def parse_pdf(source: bytes | str | Path) -> list[Block]:
    try:
        pdf = pdfplumber.open(io.BytesIO(read_source(source)))
    except Exception as exc:  # повреждённый файл
        raise ParseError(f"pdf не открывается: {exc}") from exc

    blocks: list[Block] = []
    total_chars = 0
    image_pages = 0
    with pdf:
        pages = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            total_chars += len(page.chars)
            if page.images and not page.chars:
                image_pages += 1
            items: list[Block] = []
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]
            for table in tables:
                rows = [[_cell(c) for c in row] for row in table.extract()]
                rows = [r for r in rows if any(r)]
                if rows:
                    items.append(
                        Block(
                            kind="table",
                            content="\n".join(" | ".join(r) for r in rows),
                            rows=rows,
                            metadata={"page": page_no, "top": table.bbox[1]},
                        )
                    )
            for line in page.extract_text_lines(return_chars=True):
                if any(_inside(line["top"], line["bottom"], b) for b in bboxes):
                    continue
                text = line["text"].strip()
                if not text:
                    continue
                sizes = [c["size"] for c in line["chars"] if c.get("size")]
                items.append(
                    Block(
                        kind="line",
                        content=text,
                        metadata={
                            "page": page_no,
                            "top": line["top"],
                            "font_size": round(median(sizes), 1) if sizes else None,
                        },
                    )
                )
            items.sort(key=lambda b: b.metadata.get("top") or 0.0)
            blocks.extend(items)

    if total_chars == 0:
        raise PdfNoTextLayer(
            f"нет текстового слоя: документ похож на скан ({pages} стр., "
            f"{image_pages} из них — только изображение). Распознавание (OCR) не включено, "
            "загрузите pdf с текстовым слоем."
        )
    return blocks
