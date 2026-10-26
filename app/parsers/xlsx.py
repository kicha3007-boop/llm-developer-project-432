"""Разбор xlsx: каждая таблица листа со строкой заголовков колонок.

Заголовки лежат первой непустой строкой — это обязательная часть: «800» без «мощность, Вт»
ничего не значит. Резка превращает каждую строку в текст «артикул: параметр значение».
"""

from __future__ import annotations

import io
from pathlib import Path

import openpyxl

from app.parsers import Block, ParseError, read_source


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return " ".join(str(value).split())


def parse_xlsx(source: bytes | str | Path) -> list[Block]:
    try:
        workbook = openpyxl.load_workbook(
            io.BytesIO(read_source(source)), data_only=True, read_only=True
        )
    except Exception as exc:
        raise ParseError(f"xlsx не открывается: {exc}") from exc

    blocks: list[Block] = []
    try:
        for sheet_no, sheet in enumerate(workbook.worksheets, start=1):
            rows = [[_cell(v) for v in row] for row in sheet.iter_rows(values_only=True)]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            keep = [i for i in range(width) if any(r[i] for r in rows)]  # пустые колонки — вон
            rows = [[r[i] for i in keep] for r in rows]
            meta = {"page": sheet_no, "sheet": sheet.title}
            blocks.append(Block(kind="heading", content=sheet.title, metadata={**meta, "level": 1}))
            blocks.append(
                Block(
                    kind="table",
                    content="\n".join(" | ".join(r) for r in rows),
                    rows=rows,
                    metadata=meta,
                )
            )
    finally:
        workbook.close()
    if not blocks:
        raise ParseError("в xlsx нет непустых листов")
    return blocks
