"""Разбор документов поставщиков: общие типы и выбор разборщика по расширению."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".xlsx")


class ParseError(Exception):
    """Документ не удалось разобрать; ``reason`` — внятная причина для пользователя."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class NoTextLayerError(ParseError):
    """В pdf нет текстового слоя — похоже на скан. Распознавание (OCR) не включено."""


class UnsupportedFormatError(ParseError):
    pass


@dataclass
class Block:
    """Структурная единица документа. Каждая знает свою страницу (если у формата они есть)."""

    kind: str  # line | heading | text | table
    text: str
    page: int | None = None
    section: str | None = None
    rows: list[list[str]] | None = None  # для таблиц: первая строка — заголовки колонок
    size: float | None = None  # кегль строки pdf
    top: float | None = None  # вертикальное положение строки pdf
    level: int | None = None  # уровень заголовка
    meta: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    blocks: list[Block]
    pages: int | None = None
    title: str | None = None
    fmt: str = ""


def parse_file(path: str | Path) -> ParsedDocument:
    """Разбирает файл по расширению. Нормализацию и резку делают отдельные модули."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        from app.parsers.pdf import parse_pdf

        return parse_pdf(path)
    if ext == ".docx":
        from app.parsers.docx import parse_docx

        return parse_docx(path)
    if ext == ".xlsx":
        from app.parsers.xlsx import parse_xlsx

        return parse_xlsx(path)
    raise UnsupportedFormatError(f"формат {ext or 'без расширения'} не поддерживается")
