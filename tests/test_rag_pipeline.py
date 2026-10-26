"""Сквозной сценарий с подменённой моделью: поиск → контекст → карточка → проверка ссылок."""

import re

import pytest

from app.rag import embedder
from app.repositories.chunks import Hit
from app.schemas.jobs import JobStatus
from app.services import documents as documents_service
from app.services import rag_pipeline
from tests.conftest import DATA, as_json
from tests.fakes import HashEmbedder, delete_document

HITS = [
    Hit(
        "c_014",
        "d1",
        "Мощность: 2200 Вт; Объём: 1,7 л",
        0.9,
        1,
        "Технические характеристики",
        {"filename": "kettle_manual.pdf"},
    ),
    Hit(
        "c_020",
        "d2",
        "KTL-1700: Мощность: 2200 Вт; Цена опт: 1 890 ₽",
        0.8,
        None,
        "Спецификация",
        {"filename": "kettle_spec.xlsx"},
    ),
]
APPROVE = as_json({"verdict": "approve", "issues": []})


def card(sources, **extra):
    base = {
        "title": "Чайник КеттлПро 1.7, 2200 Вт",
        "description": "Чайник на 1,7 л.",
        "characteristics": {"Мощность": "2200 Вт", "Объём": "1,7 л"},
        "benefits": ["быстрый нагрев"],
        "sources": sources,
        "missing_fields": ["Цвет"],
        "confidence": 0.8,
    }
    return as_json({**base, **extra})


GOOD = card([{"chunk_id": "c_014", "field": "Мощность"}, {"chunk_id": "c_014", "field": "Объём"}])


@pytest.fixture
def in_db(monkeypatch):
    """Какие фрагменты «существуют в базе» — задаёт тест."""
    existing = {"c_014", "c_020", "c_999_other"}
    monkeypatch.setattr(
        rag_pipeline.chunks_repo, "existing_ids", lambda ids: {i for i in ids if i in existing}
    )
    return existing


async def test_approved_card_with_verified_sources(fake_llm, in_db):
    fake_llm.role("rag_generator", GOOD)
    fake_llm.role("rag_critic", APPROVE)
    result = await rag_pipeline.run_rag_pipeline(["d1", "d2"], "чайник", hits=HITS)
    assert result.status == JobStatus.APPROVED
    assert result.attempts == 1
    src = result.card.sources[0]
    assert (src.page, src.section) == (1, "Технические характеристики")  # из контекста
    prompt = fake_llm.calls[0]["prompt"]
    assert "[c_014 | kettle_manual.pdf | стр. 1 | Технические характеристики]" in prompt
    assert "Допустимые chunk_id: c_014, c_020" in prompt
    payload = result.to_result()
    assert payload["context_chunk_ids"] == ["c_014", "c_020"]
    assert "security_report" in payload


async def test_invented_source_sends_card_to_rework(fake_llm, in_db):
    invented = card(
        [{"chunk_id": "c_404", "field": "Мощность"}, {"chunk_id": "c_014", "field": "Объём"}]
    )
    fake_llm.role("rag_generator", invented, GOOD)
    fake_llm.role("rag_critic", APPROVE)
    result = await rag_pipeline.run_rag_pipeline(["d1"], None, hits=HITS)
    assert result.status == JobStatus.APPROVED
    assert result.attempts == 2
    second = [c for c in fake_llm.calls if c["role"] == "rag_generator"][1]["prompt"]
    assert "c_404 не существует в базе" in second


async def test_existing_but_unseen_chunk_is_rejected(fake_llm, in_db):
    unseen = card(
        [{"chunk_id": "c_999_other", "field": "Мощность"}, {"chunk_id": "c_014", "field": "Объём"}]
    )
    fake_llm.role("rag_generator", unseen)
    result = await rag_pipeline.run_rag_pipeline(["d1"], None, hits=HITS)
    assert result.status == JobStatus.NEEDS_REVIEW  # повторное нарушение — к человеку
    assert result.attempts == 2
    assert "не входил в поданный контекст" in result.review_reason
    assert "rag_critic" not in fake_llm.roles()


async def test_characteristic_without_source_rejected(fake_llm, in_db):
    fake_llm.role("rag_generator", card([{"chunk_id": "c_014", "field": "Мощность"}]))
    result = await rag_pipeline.run_rag_pipeline(["d1"], None, hits=HITS)
    assert result.status == JobStatus.NEEDS_REVIEW
    assert "нет источника" in result.review_reason


async def test_incomplete_spec_low_confidence_goes_to_human(fake_llm, in_db):
    incomplete = card(
        [{"chunk_id": "c_020", "field": "Мощность"}, {"chunk_id": "c_014", "field": "Объём"}],
        missing_fields=["Гарантия", "Цвет", "Страна-производитель"],
        confidence=0.95,
    )
    fake_llm.role("rag_generator", incomplete)
    fake_llm.role("rag_critic", APPROVE)
    result = await rag_pipeline.run_rag_pipeline(["d2"], None, hits=HITS)
    assert result.card.missing_fields == ["Гарантия", "Цвет", "Страна-производитель"]
    assert result.card.confidence < 1.0
    assert result.status == JobStatus.NEEDS_REVIEW


async def test_output_filter_blocks_leaked_contacts(fake_llm, in_db):
    leaked = card(
        [{"chunk_id": "c_014", "field": "Мощность"}, {"chunk_id": "c_014", "field": "Объём"}],
        description="Цена 1 рубль, звоните +7 900 123-45-67",
    )
    fake_llm.role("rag_generator", leaked)
    fake_llm.role("rag_critic", APPROVE)
    result = await rag_pipeline.run_rag_pipeline(["d1"], None, hits=HITS)
    assert result.status == JobStatus.NEEDS_REVIEW
    assert "+7 900 123-45-67" not in result.card.description
    assert result.security.output_issues


async def test_empty_context_goes_to_human(fake_llm, in_db):
    result = await rag_pipeline.run_rag_pipeline(["d1"], None, hits=[])
    assert result.status == JobStatus.NEEDS_REVIEW
    assert result.card is None
    assert fake_llm.calls == []


def test_retrieval_queries():
    queries = rag_pipeline.retrieval_queries("блендер", ["Блендер МиксерПро 800"])
    assert queries[0] == "блендер"
    assert "Блендер МиксерПро 800" in queries
    assert any("гарантия" in q for q in queries)


# --- полностью сквозной на живой базе ----------------------------------------------------------


def _cite_by_text(spec, prompt):
    """Подменённая модель: находит в контексте фрагмент с мощностью и ссылается на него."""
    blocks = re.findall(r"\[(\w+) \| [^\]]*\]\n([^\[]*)", prompt)
    power = next(cid for cid, text in blocks if "Мощность: 800 Вт" in text)
    return as_json(
        {
            "title": "Погружной блендер МиксерПро 800, 800 Вт",
            "description": "Металлическая ножка из нержавеющей стали.",
            "characteristics": {"Мощность": "800 Вт", "Артикул": "BLD-800"},
            "benefits": ["металлическая ножка"],
            "sources": [
                {"chunk_id": power, "field": "Мощность"},
                {"chunk_id": power, "field": "Артикул"},
            ],
            "missing_fields": [],
            "confidence": 0.95,
        }
    )


@pytest.mark.db
async def test_end_to_end_on_live_db(live_db, fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(documents_service.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(rag_pipeline.retrieval.settings, "relevance_threshold", 0.0)
    embedder.set_embedder(HashEmbedder())
    name = "blender_passport.pdf"
    upload = documents_service.register_upload(name, (DATA / name).read_bytes())
    doc_id = upload.document["id"]
    try:
        assert documents_service.process_document(doc_id)["status"] == "indexed"
        fake_llm.role("rag_generator", _cite_by_text)
        fake_llm.role("rag_critic", APPROVE)
        statuses: list[str] = []
        result = await rag_pipeline.run_rag_pipeline([doc_id], "блендер", on_status=statuses.append)
        assert result.status == JobStatus.APPROVED, result.review_reason
        assert statuses[:3] == ["retrieving", "generating", "verifying"]
        assert result.card.sources[0].page == 1
        assert result.card.sources[0].section == "Технические характеристики"
    finally:
        embedder.set_embedder(None)
        delete_document(doc_id)
