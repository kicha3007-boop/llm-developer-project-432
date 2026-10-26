"""Разбор, нормализация и резка на выданных документах из data/."""

from itertools import pairwise

import pytest

from app.parsers import Block, NoTextLayerError, ParsedDocument, UnsupportedFormatError, parse_file
from app.parsers.chunker import chunk_document, split_text
from app.parsers.normalizer import clean_text, dehyphenate, normalize
from tests.conftest import DATA


def chunks_of(name: str):
    doc = normalize(parse_file(DATA / name))
    return doc, chunk_document(doc, "doc1", name)


def test_pdf_text_by_pages_and_sections():
    doc, chunks = chunks_of("blender_passport.pdf")
    assert doc.pages == 2
    assert doc.title == "Блендер погружной «МиксерПро 800»"
    pages = {c.page for c in chunks}
    assert pages == {1, 2}
    assert all(c.section for c in chunks)
    warranty = [c for c in chunks if c.section == "Гарантийные обязательства"]
    assert warranty and warranty[0].page == 2
    assert "Гарантия 24 месяца с даты продажи" in warranty[0].text


def test_two_page_document_keeps_text_but_drops_running_header():
    _doc, chunks = chunks_of("blender_passport.pdf")
    text = "\n".join(c.text for c in chunks)
    assert "supplies@technodom.example" not in text  # колонтитул «— стр. N» удалён
    assert "Металлическая ножка из нержавеющей стали" in text  # а текст на месте
    assert "Не погружать блок мотора в воду" in text


def test_pdf_spec_table_not_torn_and_has_headers():
    _doc, chunks = chunks_of("blender_passport.pdf")
    table = next(c for c in chunks if c.section == "Технические характеристики")
    assert table.kind == "table"
    assert "Параметр | Значение" in table.text
    assert "Мощность: 800 Вт" in table.text
    assert "Гарантия: 24 месяца" in table.text
    assert table.metadata["articul"] == "BLD-800"


def test_docx_sections_and_price_table_rows():
    _doc, chunks = chunks_of("blender_kp.docx")
    sections = {c.section for c in chunks}
    assert {"Прайс-лист (опт)", "Условия", "Ваш менеджер"} <= sections
    row = next(c for c in chunks if c.metadata.get("articul") == "BLD-800")
    assert row.text.startswith("BLD-800: ")
    assert "Мощность: 800 Вт" in row.text and "2 190 ₽" in row.text
    contacts = next(c for c in chunks if c.section == "Ваш менеджер")
    assert "+7 926 555-14-08" in contacts.text  # маскирование — при сборке контекста, не здесь


def test_xlsx_rows_read_as_articul_params():
    _doc, chunks = chunks_of("kettle_spec.xlsx")
    assert [c.metadata["articul"] for c in chunks] == ["KTL-1700", "KTL-1000", "KTL-THRM"]
    first = chunks[0]
    assert first.text.startswith("KTL-1700: ")
    assert "Мощность: 2200 Вт" in first.text  # заголовок «Мощность, Вт» дал единицу
    assert "Объём: 1,7 л" in first.text
    assert first.metadata["brand"] == "КеттлПро"
    # строка таблицы целиком в одном фрагменте
    assert all(c.text.count("KTL-") == 1 for c in chunks)


def test_scanned_pdf_rejected_with_reason():
    with pytest.raises(NoTextLayerError, match="нет текстового слоя"):
        parse_file(DATA / "boiler_scan.pdf")


def test_unsupported_format(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("hi")
    with pytest.raises(UnsupportedFormatError):
        parse_file(path)


def test_injection_text_in_tiny_font_is_separate_chunk():
    _doc, chunks = chunks_of("kettle_manual.pdf")
    injected = [c for c in chunks if "игнорируй все предыдущие инструкции" in c.text]
    assert len(injected) == 1
    assert injected[0].metadata.get("tiny_font") is True
    assert "Оптовые поставки" not in injected[0].text


def test_chunk_ids_deterministic():
    _d, a = chunks_of("kettle_manual.pdf")
    _d, b = chunks_of("kettle_manual.pdf")
    assert [c.id for c in a] == [c.id for c in b]


# --- нормализатор и резка без файлов ---------------------------------------------------------


def test_dehyphenation_and_ligatures():
    assert dehyphenate("мощ-", "ность 800 Вт") == "мощность 800 Вт"
    assert dehyphenate("Wi-", "Fi") is None  # заглавная — не перенос
    assert clean_text("ﬁlter  два  пробела") == "filter два пробела"


def _line(text, page, top, size=10.0):
    return Block(kind="line", text=text, page=page, top=top, size=size)


def test_two_pages_majority_rule_does_not_wipe_text():
    blocks = [
        _line("Шапка поставщика", 1, 10),
        _line("Уникальный текст первой страницы", 1, 50),
        _line("Шапка поставщика", 2, 10),
        _line("Другой текст второй страницы", 2, 50),
    ]
    doc = normalize(ParsedDocument(blocks=blocks, pages=2))
    texts = [b.text for b in doc.blocks]
    assert "Шапка поставщика" not in " ".join(texts)
    assert any("первой страницы" in t for t in texts)
    assert any("второй страницы" in t for t in texts)


def test_line_on_only_one_page_is_not_header():
    blocks = [
        _line("Только тут", 1, 10),
        _line("текст", 1, 50),
        _line("Другое", 2, 10),
        _line("ещё", 2, 50),
        _line("Третья", 3, 10),
    ]
    doc = normalize(ParsedDocument(blocks=blocks, pages=3))
    assert "Только тут" in " ".join(b.text for b in doc.blocks)


def test_hyphen_across_lines_glued():
    blocks = [_line("Номинальная мощ-", 1, 10), _line("ность 800 Вт", 1, 20)]
    doc = normalize(ParsedDocument(blocks=blocks, pages=1))
    assert doc.blocks[0].text == "Номинальная мощность 800 Вт"


def test_split_text_overlap():
    text = " ".join(f"слово{i}" for i in range(200))
    pieces = split_text(text, size=100, overlap=30)
    assert all(len(p) <= 100 for p in pieces)
    assert len(pieces) > 5
    for left, right in pairwise(pieces):
        assert left.split()[-1] in right.split()  # перекрытие: хвост повторяется в начале


def test_big_param_table_split_keeps_headers(monkeypatch):
    from app.parsers import chunker

    monkeypatch.setattr(chunker.settings, "chunk_size", 200)
    rows = [["Параметр", "Значение"]] + [[f"Параметр {i}", f"{i} ед."] for i in range(40)]
    doc = ParsedDocument(blocks=[Block(kind="table", text="", page=1, section="Хар-ки", rows=rows)])
    chunks = chunk_document(doc, "d", "f.pdf")
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("Таблица «Хар-ки» (колонки: Параметр | Значение)")
    all_lines = [ln for c in chunks for ln in c.text.splitlines()[1:]]
    assert len(all_lines) == 40  # ни одна строка не порвана и не потеряна
