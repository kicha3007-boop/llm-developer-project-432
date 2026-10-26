"""Разбор документов поставщиков: общий тип блока и выбор разборщика по формату.

Разборщики принимают байты файла (или путь) и возвращают список ``Block``. Страница, секция,
кегль и положение строки лежат в ``metadata`` блока.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_KINDS = ("pdf", "docx", "xlsx")


class ParseError(Exception):
    """Документ не удалось разобрать; ``reason`` — внятная причина для пользователя."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UnsupportedFormatError(ParseError):
    pass


@dataclass
class Block:
    """Структурная единица документа: kind — line | heading | text | table."""

    kind: str = "text"
    content: str = ""
    rows: list[list[str]] | None = None  # для таблиц: первая строка — заголовки колонок
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page(self) -> int | None:
        return self.metadata.get("page")

    @property
    def section(self) -> str | None:
        return self.metadata.get("section")


def read_source(source: bytes | str | Path) -> bytes:
    if isinstance(source, bytes | bytearray):
        return bytes(source)
    return Path(source).read_bytes()


def parse_any(data: bytes | str | Path, kind: str) -> list[Block]:
    """Разбирает файл по формату (pdf | docx | xlsx, с точкой или без)."""
    kind = kind.lower().lstrip(".")
    if kind == "pdf":
        from app.parsers.pdf import parse_pdf

        return parse_pdf(data)
    if kind == "docx":
        from app.parsers.docx import parse_docx

        return parse_docx(data)
    if kind == "xlsx":
        from app.parsers.xlsx import parse_xlsx

        return parse_xlsx(data)
    raise UnsupportedFormatError(f"формат {kind or 'без расширения'} не поддерживается")
