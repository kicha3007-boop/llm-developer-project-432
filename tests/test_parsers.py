"""Разбор, нормализация и резка на выданных документах из data/."""

from itertools import pairwise

import pytest

from app.parsers import Block, UnsupportedFormatError, parse_any
from app.parsers.chunker import chunk_blocks, split_text
from app.parsers.docx import parse_docx
from app.parsers.normalizer import clean_text, dehyphenate, guess_title, normalize
from app.parsers.pdf import PdfNoTextLayer, parse_pdf
from app.parsers.xlsx import parse_xlsx
from app.services.documents import parse_to_chunks
from tests.conftest import DATA


def read(name: str) -> bytes:
    return (DATA / name).read_bytes()


def test_pdf_text_and_tables():
    blocks = parse_pdf(read("blender_passport.pdf"))
    full = "\n".join(b.content for b in blocks)
    for probe in ("МиксерПро 800", "BLD-800", "800 Вт", "24 месяца"):
        assert probe in full
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 2
    assert tables[0].rows[0] == ["Параметр", "Значение"]
    assert len(tables[0].rows) == 13
    assert {b.page for b in blocks} == {1, 2}


def test_pdf_accepts_path_and_keeps_injection_text():
    full = "\n".join(b.content for b in parse_pdf(DATA / "kettle_manual.pdf"))
    assert "игнорируй все предыдущие инструкции" in full and "KTL-1700" in full


def test_scanned_pdf_rejected_with_reason():
    with pytest.raises(PdfNoTextLayer, match="нет текстового слоя"):
        parse_pdf(read("boiler_scan.pdf"))


def test_docx_sections_tables_and_contacts():
    blocks = parse_docx(read("blender_kp.docx"))
    full = "\n".join(b.content for b in blocks)
    for probe in ("BLD-800", "2 190", "+7 926 555-14-08"):
        assert probe in full  # маскирование — при сборке контекста, не при разборе
    assert len([b for b in blocks if b.kind == "table"]) == 1
    assert any(b.kind == "heading" and b.content == "Ваш менеджер" for b in blocks)


def test_xlsx_rows_become_chunks_with_articul():
    chunks = parse_to_chunks(read("kettle_spec.xlsx"), "xlsx", "doc1")
    assert [c["metadata"].get("articul") for c in chunks] == ["KTL-1700", "KTL-1000", "KTL-THRM"]
    first = chunks[0]
    assert first["text"].startswith("KTL-1700: ")
    assert "Мощность: 2200 Вт" in first["text"]  # заголовок «Мощность, Вт» дал единицу
    assert first["metadata"]["brand"] == "КеттлПро"
    assert all(c["text"].count("KTL-") == 1 for c in chunks)  # строка не разорвана


def test_xlsx_header_row_kept():
    table = next(b for b in parse_xlsx(read("kettle_spec.xlsx")) if b.kind == "table")
    assert table.rows[0][:2] == ["Артикул", "Бренд"]


def test_two_page_pdf_drops_running_footer_but_keeps_text():
    full = "\n".join(b.content for b in normalize(parse_pdf(read("blender_passport.pdf"))))
    assert "стр. 1" not in full and "стр. 2" not in full
    assert "supplies@technodom.example" not in full
    assert "Металлическая ножка из нержавеющей стали" in full
    assert "Не погружать блок мотора в воду" in full


def test_chunk_metadata_and_table_layout():
    blocks = normalize(parse_pdf(read("blender_passport.pdf")))
    chunks = chunk_blocks(blocks, "docX", "blender_passport.pdf")
    assert all(c["metadata"]["doc_id"] == "docX" for c in chunks)
    assert all("page" in c["metadata"] and c["metadata"]["section"] for c in chunks)
    table = next(c for c in chunks if c["metadata"]["section"] == "Технические характеристики")
    assert "Параметр | Значение" in table["text"]
    assert "Мощность: 800 Вт" in table["text"] and "Гарантия: 24 месяца" in table["text"]
    assert table["metadata"]["articul"] == "BLD-800"
    assert guess_title(blocks) == "Блендер погружной «МиксерПро 800»"


def test_tiny_font_injection_is_separate_chunk():
    chunks = parse_to_chunks(read("kettle_manual.pdf"), "pdf", "k")
    injected = [c for c in chunks if "игнорируй все предыдущие инструкции" in c["text"]]
    assert len(injected) == 1
    assert injected[0]["metadata"].get("tiny_font") is True
    assert "Оптовые поставки" not in injected[0]["text"]


def test_chunk_ids_deterministic():
    a = parse_to_chunks(read("kettle_manual.pdf"), "pdf", "k")
    b = parse_to_chunks(read("kettle_manual.pdf"), "pdf", "k")
    assert [c["id"] for c in a] == [c["id"] for c in b]


def test_unsupported_format():
    with pytest.raises(UnsupportedFormatError):
        parse_any(b"hello", "txt")


# --- нормализатор и резка без файлов ---------------------------------------------------------


def test_clean_text():
    assert clean_text("мощ-\nность") == "мощность"
    assert clean_text("слева  справа") == "слева справа"
    assert clean_text("ﬁlter  два") == "filter два"
    assert dehyphenate("мощ-", "ность 800 Вт") == "мощность 800 Вт"
    assert dehyphenate("Wi-", "Fi") is None  # заглавная — не перенос


def _line(text, page, top, size=10.0):
    return Block(kind="line", content=text, metadata={"page": page, "top": top, "font_size": size})


def test_two_pages_majority_rule_does_not_wipe_text():
    blocks = [
        _line("Шапка поставщика", 1, 10),
        _line("Уникальный текст первой страницы", 1, 50),
        _line("Шапка поставщика", 2, 10),
        _line("Другой текст второй страницы", 2, 50),
    ]
    texts = " ".join(b.content for b in normalize(blocks))
    assert "Шапка поставщика" not in texts
    assert "первой страницы" in texts and "второй страницы" in texts


def test_line_on_one_page_only_is_not_header():
    blocks = [
        _line("Только тут", 1, 10),
        _line("текст", 1, 50),
        _line("Другое", 2, 10),
        _line("ещё", 2, 50),
        _line("Третья", 3, 10),
    ]
    assert "Только тут" in " ".join(b.content for b in normalize(blocks))


def test_hyphen_across_lines_glued():
    blocks = normalize([_line("Номинальная мощ-", 1, 10), _line("ность 800 Вт", 1, 20)])
    assert blocks[0].content == "Номинальная мощность 800 Вт"


def test_text_blocks_of_one_section_merge():
    blocks = [
        Block(content="Первый абзац", metadata={"page": 2, "section": "Правила"}),
        Block(content="Второй абзац", metadata={"page": 2, "section": "Правила"}),
    ]
    chunks = chunk_blocks(blocks, "doc1")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page"] == 2 and chunks[0]["metadata"]["section"] == "Правила"


def test_spec_table_row_per_chunk():
    rows = [["Артикул", "Мощность, Вт"], ["KTL-1700", "2200"], ["KTL-1000", "1800"]]
    chunks = chunk_blocks([Block(kind="table", rows=rows, metadata={"page": 1})], "doc2")
    assert [c["metadata"]["articul"] for c in chunks] == ["KTL-1700", "KTL-1000"]
    assert chunks[0]["text"] == "KTL-1700: Мощность: 2200 Вт"


def test_split_text_overlap():
    text = " ".join(f"слово{i}" for i in range(200))
    pieces = split_text(text, size=100, overlap=30)
    assert all(len(p) <= 100 for p in pieces) and len(pieces) > 5
    for left, right in pairwise(pieces):
        assert left.split()[-1] in right.split()


def test_big_param_table_split_keeps_headers(monkeypatch):
    from app.parsers import chunker

    monkeypatch.setattr(chunker.settings, "chunk_size", 200)
    rows = [["Параметр", "Значение"]] + [[f"Параметр {i}", f"{i} ед."] for i in range(40)]
    chunks = chunk_blocks(
        [Block(kind="table", rows=rows, metadata={"page": 1, "section": "Хар-ки"})], "d"
    )
    assert len(chunks) > 1
    assert all(
        c["text"].startswith("Таблица «Хар-ки» (колонки: Параметр | Значение)") for c in chunks
    )
    assert sum(len(c["text"].splitlines()) - 1 for c in chunks) == 40
