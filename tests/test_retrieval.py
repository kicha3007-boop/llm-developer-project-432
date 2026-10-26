"""Поиск: слияние рангов и порог — на заранее заданных векторах, без загрузки модели."""

from dataclasses import dataclass, field

import pytest
from psycopg.types.json import Jsonb

from app.core.db import connection
from app.rag.retrieval import rrf_fuse
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.repositories.chunks import SearchFilters, query_tokens
from tests.fakes import delete_document, unit

# --- RRF на Python (эталон формулы) ------------------------------------------------------------


def test_rrf_item_found_by_both_lists_ranks_higher():
    vector = ["c14", "c07", "c21"]
    text = ["c07", "c14", "c02"]
    fused = [cid for cid, _ in rrf_fuse([vector, text], k=60)]
    assert set(fused) == {"c14", "c07", "c21", "c02"}  # объединение
    assert fused[:2] == ["c07", "c14"] or fused[:2] == ["c14", "c07"]
    assert fused.index("c21") > 1 and fused.index("c02") > 1


def test_rrf_uses_positions_not_scores():
    scores = dict(rrf_fuse([["a", "b"], ["b"]], k=60))
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)


def test_query_tokens_drop_question_words():
    assert query_tokens("Какая мощность у KTL-1700?") == ["мощность", "ktl-1700"]


# --- SQL-поиск на живой базе -------------------------------------------------------------------


@dataclass
class _Chunk:
    id: str
    ord: int
    text: str
    kind: str = "text"
    page: int | None = 1
    section: str | None = "Тест"
    metadata: dict = field(default_factory=dict)


@pytest.fixture
def corpus(live_db, unique):
    """Три фрагмента с векторами, заданными руками."""
    doc_id = f"t{unique}"
    documents_repo.get_or_create(
        document_id=doc_id,
        sha256=f"sha-{unique}",
        filename="t.pdf",
        extension=".pdf",
        size_bytes=1,
        storage_path="t.pdf",
    )
    items = [
        (
            _Chunk(f"{unique}a", 0, "Мотор блендера 800 Вт", metadata={"articul": "BLD-800"}),
            unit(1),
        ),
        (
            _Chunk(f"{unique}b", 1, "Артикул KTL-1700 чайник", metadata={"articul": "KTL-1700"}),
            unit(2),
        ),
        (_Chunk(f"{unique}c", 2, "Правила ухода за изделием"), unit(3)),
    ]
    chunks_repo.replace_for_document(doc_id, [c for c, _ in items])
    chunks_repo.save_embeddings([(c.id, v) for c, v in items], "test")
    yield doc_id, {c.id[len(unique) :]: c.id for c, _ in items}
    delete_document(doc_id)


@pytest.mark.db
def test_vector_search_with_threshold(corpus):
    doc_id, ids = corpus
    query = unit(1, 3, weights=[0.9, 0.3])
    flt = SearchFilters(document_ids=[doc_id])
    hits = chunks_repo.vector_search(query, top_k=5, threshold=0.0, filters=flt)
    assert [h.chunk_id for h in hits][:2] == [ids["a"], ids["c"]]
    strict = chunks_repo.vector_search(query, top_k=5, threshold=0.8, filters=flt)
    assert [h.chunk_id for h in strict] == [ids["a"]]  # порог отсёк слабые


@pytest.mark.db
def test_text_search_finds_articul_exactly(corpus):
    doc_id, ids = corpus
    hits = chunks_repo.text_search(
        "KTL-1700", top_k=5, filters=SearchFilters(document_ids=[doc_id])
    )
    assert [h.chunk_id for h in hits] == [ids["b"]]


@pytest.mark.db
def test_hybrid_returns_union_and_fuses_by_rank(corpus):
    doc_id, ids = corpus
    flt = SearchFilters(document_ids=[doc_id])
    # по смыслу запрос ближе к «a», по словам совпадает только «b»
    hits = chunks_repo.hybrid_search(
        "KTL-1700", unit(1), top_k=5, threshold=0.5, rrf_k=60, filters=flt
    )
    got = [h.chunk_id for h in hits]
    assert set(got) == {ids["a"], ids["b"]}  # «c» отсёк порог векторной ветки
    # фрагмент, найденный обоими поисками, выше найденного одним
    both = chunks_repo.hybrid_search(
        "мотор 800", unit(1), top_k=5, threshold=0.5, rrf_k=60, filters=flt
    )
    assert both[0].chunk_id == ids["a"]
    assert both[0].score == pytest.approx(1 / 61 + 1 / 61)


@pytest.mark.db
def test_metadata_filter_by_articul(corpus):
    doc_id, ids = corpus
    flt = SearchFilters(document_ids=[doc_id], articul="KTL-1700")
    hits = chunks_repo.vector_search(unit(1), top_k=5, threshold=0.0, filters=flt)
    assert [h.chunk_id for h in hits] == [ids["b"]]


@pytest.mark.db
def test_save_embeddings_batch_and_missing(corpus):
    doc_id, _ids = corpus
    assert chunks_repo.missing_embeddings(10, document_id=doc_id) == []
    with connection() as conn:
        conn.execute("UPDATE chunks SET embedding = NULL WHERE document_id = %s", (doc_id,))
        conn.execute("UPDATE chunks SET metadata = %s WHERE document_id = %s", (Jsonb({}), doc_id))
    assert len(chunks_repo.missing_embeddings(10, document_id=doc_id)) == 3
