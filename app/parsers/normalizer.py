"""Нормализация разобранного документа: колонтитулы, переносы, лигатуры, пробелы, секции.

Колонтитул — строка, повторяющаяся у края страниц. Правило «повторяется на большинстве страниц»
на двухстраничном документе опасно: «большинство» от двух — одна страница, и такое правило
выметает весь текст. Поэтому здесь колонтитулом считается только строка у края страницы
(первые/последние строки), которая повторяется минимум на ДВУХ страницах и больше чем на половине
(цифры при сравнении не различаются: «стр. 1» и «стр. 2» — один колонтитул). На одностраничном
документе по повторам ничего не удаляется — только краевые строки с номером страницы.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

from app.parsers import Block, ParsedDocument

LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
EDGE_LINES = 2
_PAGE_NUMBER_RE = re.compile(
    r"(стр\.?\s*\d+|страница\s+\d+|page\s+\d+|^\s*\d+\s*(/|из)\s*\d+\s*$|^\s*-?\s*\d+\s*-?\s*$)",
    re.IGNORECASE,
)
_SPACES_RE = re.compile(r"[ \t   ]+")
_HYPHEN_END_RE = re.compile(r"(\w)[-­‐]$")


def clean_text(text: str) -> str:
    """Лигатуры, мягкие переносы, неразрывные пробелы и повторные пробелы."""
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.replace("­", "").replace("​", "")
    text = unicodedata.normalize("NFC", text)
    return _SPACES_RE.sub(" ", text).strip()


def _key(text: str) -> str:
    return re.sub(r"\d+", "#", clean_text(text).lower())


def find_running_lines(blocks: list[Block]) -> set[tuple[int, int]]:
    """Индексы (страница, номер блока) строк-колонтитулов."""
    by_page: dict[int, list[int]] = defaultdict(list)
    for idx, block in enumerate(blocks):
        if block.kind == "line" and block.page is not None:
            by_page[block.page].append(idx)
    pages = len(by_page)
    edge: dict[int, list[int]] = {
        page: idxs[:EDGE_LINES] + idxs[-EDGE_LINES:] for page, idxs in by_page.items()
    }
    counts: Counter[str] = Counter()
    for idxs in edge.values():
        for key in {_key(blocks[i].text) for i in idxs}:
            counts[key] += 1
    need = max(2, math.floor(pages / 2) + 1)
    running: set[tuple[int, int]] = set()
    for page, idxs in edge.items():
        for i in idxs:
            text = blocks[i].text
            if (pages >= 2 and counts[_key(text)] >= need) or _PAGE_NUMBER_RE.search(text):
                running.add((page, i))
    return running


def dehyphenate(prev: str, nxt: str) -> str | None:
    """«мощ-» + «ность» → «мощность». None — переноса нет."""
    if _HYPHEN_END_RE.search(prev) and nxt[:1].islower():
        return prev[:-1] + nxt
    return None


def _body_size(blocks: list[Block]) -> float | None:
    weights: Counter[float] = Counter()
    for b in blocks:
        if b.kind == "line" and b.size:
            weights[b.size] += len(b.text)
    return weights.most_common(1)[0][0] if weights else None


def _merge_lines(blocks: list[Block]) -> list[Block]:
    """Строки pdf → заголовки и параграфы. Абзац рвётся на заголовке, смене кегля и странице."""
    body = _body_size(blocks)
    out: list[Block] = []
    current: Block | None = None
    for block in blocks:
        if block.kind != "line":
            current = None
            out.append(block)
            continue
        text = clean_text(block.text)
        if not text:
            continue
        is_heading = bool(body and block.size and block.size >= body + 1.5)
        if is_heading:
            current = None
            out.append(Block(kind="heading", text=text, page=block.page, size=block.size, level=1))
            continue
        same_par = (
            current is not None
            and current.page == block.page
            and (current.size or 0) == (block.size or 0)
        )
        if same_par and current is not None:
            joined = dehyphenate(current.text, text)
            current.text = joined if joined is not None else f"{current.text} {text}"
        else:
            tiny = bool(body and block.size and block.size < body * 0.75)
            current = Block(
                kind="text",
                text=text,
                page=block.page,
                size=block.size,
                meta={"tiny_font": True} if tiny else {},
            )
            out.append(current)
    return out


def normalize(doc: ParsedDocument) -> ParsedDocument:
    """Чистит блоки документа и проставляет секции."""
    blocks = doc.blocks
    if any(b.kind == "line" for b in blocks):
        running = find_running_lines(blocks)
        blocks = [b for i, b in enumerate(blocks) if (b.page or 0, i) not in running]
        blocks = _merge_lines(blocks)

    result: list[Block] = []
    section: str | None = None
    for block in blocks:
        if block.kind == "table":
            rows = [[clean_text(c) for c in row] for row in block.rows or []]
            result.append(
                Block(
                    kind="table",
                    text="",
                    page=block.page,
                    section=section,
                    rows=rows,
                    meta=dict(block.meta),
                )
            )
            continue
        text = clean_text(block.text)
        if not text:
            continue
        if block.kind == "heading":
            section = text
            result.append(
                Block(kind="heading", text=text, page=block.page, section=text, level=block.level)
            )
            continue
        result.append(
            Block(
                kind="text",
                text=text,
                page=block.page,
                section=section,
                size=block.size,
                meta=dict(block.meta),
            )
        )
    title = clean_text(doc.title) if doc.title else None
    return ParsedDocument(blocks=result, pages=doc.pages, title=title, fmt=doc.fmt)
