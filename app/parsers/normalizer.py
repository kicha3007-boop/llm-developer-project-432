"""Нормализация разобранного документа: колонтитулы, переносы, лигатуры, пробелы, секции.

Колонтитул — строка, повторяющаяся у края страниц. Правило «повторяется на большинстве страниц»
на двухстраничном документе опасно: «большинство» от двух — одна страница, и такое правило
выметает весь текст. Здесь колонтитулом считается только строка у края страницы (первые/последние
строки), которая повторяется минимум на ДВУХ страницах и больше чем на половине (цифры при
сравнении не различаются: «стр. 1» и «стр. 2» — один колонтитул). На одностраничном документе по
повторам ничего не удаляется — только краевые строки с номером страницы.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

from app.parsers import Block

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
_HYPHEN_BREAK_RE = re.compile(r"(\w)[-­‐]\s*\n\s*([a-zа-яё])")
_HYPHEN_END_RE = re.compile(r"(\w)[-­‐]$")


def clean_text(text: str) -> str:
    """Перенос через дефис склеивается, лигатуры и неразрывные пробелы заменяются,
    повторные пробелы схлопываются."""
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.replace("​", "")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("­", "")
    text = unicodedata.normalize("NFC", text)
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _key(text: str) -> str:
    return re.sub(r"\d+", "#", clean_text(text).lower())


def find_running_lines(blocks: list[Block]) -> set[int]:
    """Индексы блоков-строк, которые являются колонтитулами."""
    by_page: dict[int, list[int]] = defaultdict(list)
    for idx, block in enumerate(blocks):
        if block.kind == "line" and block.page is not None:
            by_page[block.page].append(idx)
    pages = len(by_page)
    edge = {page: idxs[:EDGE_LINES] + idxs[-EDGE_LINES:] for page, idxs in by_page.items()}
    counts: Counter[str] = Counter()
    for idxs in edge.values():
        for key in {_key(blocks[i].content) for i in idxs}:
            counts[key] += 1
    need = max(2, math.floor(pages / 2) + 1)
    running: set[int] = set()
    for idxs in edge.values():
        for i in idxs:
            text = blocks[i].content
            if (pages >= 2 and counts[_key(text)] >= need) or _PAGE_NUMBER_RE.search(text):
                running.add(i)
    return running


def dehyphenate(prev: str, nxt: str) -> str | None:
    """«мощ-» + «ность» → «мощность». None — переноса нет."""
    if _HYPHEN_END_RE.search(prev) and nxt[:1].islower():
        return prev[:-1] + nxt
    return None


def _body_size(blocks: list[Block]) -> float | None:
    weights: Counter[float] = Counter()
    for b in blocks:
        size = b.metadata.get("font_size")
        if b.kind == "line" and size:
            weights[size] += len(b.content)
    return weights.most_common(1)[0][0] if weights else None


def _merge_lines(blocks: list[Block]) -> list[Block]:
    """Строки pdf → заголовки и абзацы. Абзац рвётся на заголовке, смене кегля и странице."""
    body = _body_size(blocks)
    out: list[Block] = []
    current: Block | None = None
    for block in blocks:
        if block.kind != "line":
            current = None
            out.append(block)
            continue
        text = clean_text(block.content)
        if not text:
            continue
        size = block.metadata.get("font_size")
        meta = {"page": block.page, "font_size": size}
        if body and size and size >= body + 1.5:
            current = None
            out.append(Block(kind="heading", content=text, metadata={**meta, "level": 1}))
            continue
        same = (
            current is not None
            and current.page == block.page
            and current.metadata.get("font_size") == size
        )
        if same and current is not None:
            joined = dehyphenate(current.content, text)
            current.content = joined if joined is not None else f"{current.content} {text}"
        else:
            if body and size and size < body * 0.75:
                meta["tiny_font"] = True
            current = Block(kind="text", content=text, metadata=meta)
            out.append(current)
    return out


def normalize(blocks: list[Block]) -> list[Block]:
    """Чистит блоки документа и проставляет секции (последний заголовок перед блоком)."""
    if any(b.kind == "line" for b in blocks):
        running = find_running_lines(blocks)
        blocks = _merge_lines([b for i, b in enumerate(blocks) if i not in running])

    result: list[Block] = []
    section: str | None = None
    for block in blocks:
        meta = dict(block.metadata)
        if block.kind == "table":
            rows = [[clean_text(c) for c in row] for row in block.rows or []]
            meta.setdefault("section", section)
            result.append(
                Block(
                    kind="table",
                    content="\n".join(" | ".join(r) for r in rows),
                    rows=rows,
                    metadata=meta,
                )
            )
            continue
        text = clean_text(block.content)
        if not text:
            continue
        if block.kind == "heading":
            section = text
            result.append(Block(kind="heading", content=text, metadata={**meta, "section": text}))
            continue
        meta.setdefault("section", section)
        result.append(Block(kind="text", content=text, metadata=meta))
    return result


def guess_title(blocks: list[Block]) -> str | None:
    """Заголовок документа для поиска: крупный заголовок первой страницы (надпись капсом вроде
    «ПАСПОРТ ИЗДЕЛИЯ» пропускается) или первый заголовок второго уровня (docx)."""
    headings = [b for b in blocks if b.kind == "heading"]
    sized = sorted(
        (b for b in headings if b.page in (None, 1) and b.metadata.get("font_size")),
        key=lambda b: -(b.metadata.get("font_size") or 0),
    )
    for block in sized:
        if not block.content.isupper():
            return block.content
    for block in headings:
        if (block.metadata.get("level") or 0) >= 2:
            return block.content
    return headings[0].content if headings else None
