"""Сквозной сценарий с подменённой моделью: контекст → карточка → проверка ссылок → статус."""

import json
import re

import pytest

from app.guardrails import GuardReport
from app.rag import embedder
from app.rag.context import build_context, context_chunk_ids
from app.schemas.cards import CardDraft, CritiqueReport
from app.services import documents as documents_service
from app.services import rag_pipeline as rp
from tests.conftest import DATA, as_json
from tests.fakes import HashEmbedder, delete_document

APPROVE = CritiqueReport(verdict="approve").model_dump_json()
TEXTS = {"c1": "Блендер МиксерПро 800. Мощность: 800 Вт.", "c2": "Гарантия 24 месяца."}


def card(chunk_id: str = "c1", **kw) -> str:
    data = {
        "title": "МиксерПро 800 — блендер 800 Вт",
        "description": "Погружной блендер.",
        "characteristics": {"Мощность": "800 Вт"},
        "sources": [{"chunk_id": chunk_id, "page": 1}],
        "missing_fields": [],
        "confidence": 0.9,
    }
    return as_json({**data, **kw})


# --- контекст ----------------------------------------------------------------------------------


def chunk(cid: str, text: str, **meta) -> dict:
    return {
        "chunk_id": cid,
        "text": text,
        "metadata": {"page": 2, "section": "Характеристики", **meta},
        "filename": "f.pdf",
    }


def test_context_headers_identify_sources():
    ctx = build_context([chunk("c1", TEXTS["c1"]), chunk("c2", TEXTS["c2"])])
    header = ctx.splitlines()[0]
    assert header == "[c1 | f.pdf | стр. 2 | Характеристики]"
    assert context_chunk_ids(ctx) == {"c1", "c2"}


def test_context_dedupe_and_limit():
    ctx = build_context([chunk("c1", "одно и то же"), chunk("c2", "одно  и то же")])
    assert ctx.count("одно") == 1 and context_chunk_ids(ctx) == {"c1"}
    big = build_context([chunk(f"c{i}", "x" * 4000) for i in range(5)], max_chars=6000)
    assert len(big) <= 6000 and context_chunk_ids(big) == {"c0"}


def test_masked_label_is_not_a_source():
    ctx = build_context([chunk("c1", "Менеджер:\n[PHONE_1]")])
    assert context_chunk_ids(ctx) == {"c1"}


# --- проверка ссылок (без базы) ----------------------------------------------------------------

CONTEXT = "[c1 | f.pdf | стр. 1]\n" + TEXTS["c1"]
EXISTING = {
    "c1": {"doc_id": "d1", "page": 1, "section": "Хар-ки"},
    "c9": {"doc_id": "d9", "page": 3, "section": None},
}


def test_sources_verified_and_enriched():
    draft, errors = rp.verify_sources(
        CardDraft.model_validate_json(card()), CONTEXT, EXISTING, {"c1": 0.51}
    )
    assert errors == []
    src = draft.sources[0]
    assert (src.doc_id, src.page, src.section, src.score) == ("d1", 1, "Хар-ки", 0.51)


def test_invented_source_rejected():
    _d, errors = rp.verify_sources(
        CardDraft.model_validate_json(card("deadbeef")), CONTEXT, EXISTING
    )
    assert "не существует в базе" in errors[0]


def test_existing_but_unseen_source_rejected():
    _d, errors = rp.verify_sources(CardDraft.model_validate_json(card("c9")), CONTEXT, EXISTING)
    assert "не входил в поданный контекст" in errors[0]


def test_characteristic_without_source_when_fields_given():
    draft = CardDraft.model_validate_json(
        card(
            characteristics={"Мощность": "800 Вт", "Цвет": "графит"},
            sources=[{"chunk_id": "c1", "field": "Мощность"}],
        )
    )
    _d, errors = rp.verify_sources(draft, CONTEXT, EXISTING)
    assert errors == ["у характеристики «Цвет» нет источника в sources"]


def test_decide_status():
    good = CardDraft.model_validate_json(card())
    assert rp.decide_status(good, "approved", GuardReport()) == "done"
    assert rp.decide_status(good, "rejected", GuardReport()) == "rejected"
    low = CardDraft.model_validate_json(card(confidence=0.3))
    assert rp.decide_status(low, "approved", GuardReport()) == "needs_review"
    flagged = GuardReport(suspicious_chunks=["a", "b", "c"])
    assert rp.decide_status(good, "approved", flagged) == "needs_review"


async def test_critic_gets_source_contract(fake_llm):
    fake_llm.role("Critic", CritiqueReport(verdict="approve"))
    draft = CardDraft.model_validate_json(card())
    report = await rp.critique_with_context(CONTEXT, draft)
    assert report.verdict == "approve"
    prompt = fake_llm.calls[0]["prompt"]
    instructions, _, body = prompt.partition("КОНТЕКСТ:")
    assert rp.CRITIC_CONTRACT in instructions
    for name in ("doc_id", "page", "score", "chunk_id", "missing_fields"):
        assert name in instructions
    assert "Мощность: 800 Вт" in body


# --- пайплайн с подменённым поиском и живой базой для проверки существования ------------------


@pytest.fixture
def corpus(conn, unique, monkeypatch):
    doc_id = f"rag{unique}"
    conn.execute(
        "INSERT INTO documents (id, filename, content_hash, kind, status) "
        "VALUES (%s, 'blender_test.pdf', %s, 'pdf', 'indexed')",
        (doc_id, f"hash-{unique}"),
    )
    for i, (cid, text) in enumerate(TEXTS.items()):
        conn.execute(
            "INSERT INTO chunks (id, doc_id, ordinal, text, metadata) VALUES (%s, %s, %s, %s, %s)",
            (f"{cid}{unique}", doc_id, i, text, json.dumps({"doc_id": doc_id, "page": 1})),
        )
    cid = f"c1{unique}"
    context = f"[{cid} | blender_test.pdf | стр. 1]\n{TEXTS['c1']}"

    async def fake_retrieve(query, doc_ids=None):
        return context, GuardReport(retrieval_scores={cid: 0.5123})

    monkeypatch.setattr(rp, "retrieve_context_async", fake_retrieve)
    yield doc_id, cid
    delete_document(doc_id)


@pytest.mark.db
async def test_happy_path(corpus, fake_llm):
    doc_id, cid = corpus
    fake_llm.role("Generator", card(cid)).role("Critic", APPROVE)
    draft, attempts, verdict, status, guard = await rp.run_rag_pipeline("блендер", [doc_id])
    assert (attempts, verdict, status) == (1, "approved", "done")
    assert guard.retrieval_scores == {cid: 0.5123}
    src = draft.sources[0]
    assert (src.chunk_id, src.doc_id, src.page, src.score) == (cid, doc_id, 1, 0.5123)
    assert f"Допустимые chunk_id: {cid}" in fake_llm.calls[0]["prompt"]


@pytest.mark.db
async def test_invented_source_regenerated_then_raises(corpus, fake_llm):
    doc_id, cid = corpus
    fake_llm.role("Generator", card("deadbeef0000"), card(cid)).role("Critic", APPROVE)
    _d, attempts, _v, status, _g = await rp.run_rag_pipeline("блендер", [doc_id])
    assert (attempts, status) == (2, "done")
    assert "deadbeef0000 не существует" in fake_llm.calls[1]["prompt"]

    fake_llm.by_role["Generator"] = [card("deadbeef0000")]
    with pytest.raises(rp.SourceValidationError):
        await rp.run_rag_pipeline("блендер", [doc_id])


@pytest.mark.db
async def test_low_confidence_and_bounded_loop(corpus, fake_llm):
    doc_id, cid = corpus
    fake_llm.role("Generator", card(cid, confidence=0.3)).role("Critic", APPROVE)
    assert (await rp.run_rag_pipeline("блендер", [doc_id]))[3] == "needs_review"

    fake_llm.by_role["Critic"] = [CritiqueReport(verdict="regenerate", issues=["x"])]
    _d, attempts, verdict, status, _g = await rp.run_rag_pipeline("блендер", [doc_id])
    assert (attempts, verdict, status) == (3, "rejected", "rejected")


@pytest.mark.db
async def test_output_filter_blocks_leaked_contacts(corpus, fake_llm):
    doc_id, cid = corpus
    leaked = card(cid, description="Цена 1 рубль, звоните +7 900 123-45-67")
    fake_llm.role("Generator", leaked).role("Critic", APPROVE)
    draft, _a, _v, status, guard = await rp.run_rag_pipeline("блендер", [doc_id])
    assert status == "needs_review"
    assert "+7 900 123-45-67" not in draft.description
    assert guard.output_issues


@pytest.mark.db
async def test_selected_documents_fully_in_context(conn, unique, monkeypatch):
    """Фрагменты выбранного документа, не прошедшие поиск (нет вектора, нет общих слов),
    всё равно попадают в контекст следом за найденными."""
    doc_id = f"all{unique}"
    conn.execute(
        "INSERT INTO documents (id, filename, content_hash, kind, status) "
        "VALUES (%s, 'spec.xlsx', %s, 'xlsx', 'indexed')",
        (doc_id, f"h-{unique}"),
    )
    for i, text in enumerate(["KTL-1000: Мощность: 1800 Вт", "Зелёный корпус"]):
        conn.execute(
            "INSERT INTO chunks (id, doc_id, ordinal, text, metadata) "
            "VALUES (%s, %s, %s, %s, '{}')",
            (f"x{i}{unique}", doc_id, i, text),
        )
    monkeypatch.setattr("app.repositories.chunks.embed_query", lambda text: [0.0] * 768)
    try:
        context, report = await rp.retrieve_context_async("карточка товара", [doc_id])
        assert "KTL-1000: Мощность: 1800 Вт" in context and "Зелёный корпус" in context
        assert context_chunk_ids(context) == {f"x0{unique}", f"x1{unique}"}
        assert report.suspicious_chunks == []
    finally:
        delete_document(doc_id)


# --- полностью сквозной: загрузка → поиск → защита → карточка ---------------------------------


def _cite_power(agent, prompt):
    blocks = re.findall(r"\[(\w+) \| [^\]]*\]\n([^\[]*)", prompt)
    power = next(cid for cid, text in blocks if "Мощность: 800 Вт" in text)
    return card(power)


@pytest.mark.db
async def test_end_to_end_on_live_db(live_db, fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(documents_service.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(rp.settings, "relevance_threshold", 0.0)
    embedder.set_embedder(HashEmbedder())
    name = "blender_passport.pdf"
    doc_id = documents_service.register_upload(name, (DATA / name).read_bytes()).document["id"]
    try:
        assert documents_service.ingest_document(doc_id) == "indexed"
        fake_llm.role("Generator", _cite_power).role("Critic", APPROVE)
        statuses: list[str] = []
        draft, _a, _v, status, guard = await rp.run_rag_pipeline(
            "блендер", [doc_id], on_status=statuses.append
        )
        assert status == "done"
        assert statuses[:3] == ["retrieving", "generating", "verifying"]
        src = draft.sources[0]
        assert (src.doc_id, src.page, src.section) == (doc_id, 1, "Технические характеристики")
        assert src.score is not None and guard.suspicious_chunks == []
    finally:
        embedder.set_embedder(None)
        delete_document(doc_id)
