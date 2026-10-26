"""Поиск: слияние рангов и порог — на заранее заданных векторах, без загрузки модели."""

import pytest

from app.rag.retrieval import rrf_fuse
from app.repositories import chunks as chunks_repo
from app.repositories.chunks import query_tokens, vector_literal
from tests.fakes import delete_document, unit

# --- RRF на Python (эталон формулы) ------------------------------------------------------------


def test_rrf_item_found_by_both_lists_ranks_higher():
    fused = [cid for cid, _ in rrf_fuse([["c14", "c07", "c21"], ["c07", "c14", "c02"]], k=60)]
    assert set(fused) == {"c14", "c07", "c21", "c02"}  # объединение
    assert set(fused[:2]) == {"c07", "c14"}


def test_rrf_uses_positions_not_scores():
    scores = dict(rrf_fuse([["a", "b"], ["b"]], k=60))
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)


def test_query_tokens_drop_question_words():
    assert query_tokens("Какая мощность у KTL-1700?") == ["мощность", "ktl-1700"]


# --- SQL-поиск на живой базе -------------------------------------------------------------------


@pytest.fixture
def corpus(conn, unique, monkeypatch):
    """Три фрагмента с векторами, заданными руками; вектор запроса тоже задаёт тест."""
    doc_id = f"t{unique}"
    conn.execute(
        "INSERT INTO documents (id, filename, content_hash, kind, status) "
        "VALUES (%s, 't.pdf', %s, 'pdf', 'indexed')",
        (doc_id, f"hash-{unique}"),
    )
    rows = [
        (f"{unique}a", "Мотор блендера 800 Вт", '{"articul": "BLD-800", "page": 1}', unit(1)),
        (f"{unique}b", "Артикул KTL-1700 чайник", '{"articul": "KTL-1700", "page": 1}', unit(2)),
        (f"{unique}c", "Правила ухода за изделием", '{"page": 2}', unit(3)),
    ]
    for i, (cid, text, meta, vec) in enumerate(rows):
        conn.execute(
            "INSERT INTO chunks (id, doc_id, ordinal, text, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s::vector)",
            (cid, doc_id, i, text, meta, vector_literal(vec)),
        )
    query_vec = {"value": unit(1)}
    monkeypatch.setattr(chunks_repo, "embed_query", lambda text: query_vec["value"])
    yield doc_id, {cid[len(unique) :]: cid for cid, *_ in rows}, query_vec
    delete_document(doc_id)


@pytest.mark.db
def test_vector_search_with_threshold(conn, corpus):
    doc_id, ids, q = corpus
    q["value"] = unit(1, 3, weights=[0.9, 0.3])
    hits = chunks_repo.search_vector(conn, "мотор", 5, 0.0, [doc_id])
    assert [h["chunk_id"] for h in hits][:2] == [ids["a"], ids["c"]]
    assert hits[0]["page"] == 1 and hits[0]["filename"] == "t.pdf"
    strict = chunks_repo.search_vector(conn, "мотор", 5, 0.8, [doc_id])
    assert [h["chunk_id"] for h in strict] == [ids["a"]]  # порог отсёк слабые


@pytest.mark.db
def test_fts_finds_articul_exactly(conn, corpus):
    doc_id, ids, _q = corpus
    hits = chunks_repo.search_fts(conn, "KTL-1700", 5, [doc_id])
    assert [h["chunk_id"] for h in hits] == [ids["b"]]


@pytest.mark.db
def test_hybrid_returns_union_and_fuses_by_rank(conn, corpus):
    doc_id, ids, _q = corpus
    # по смыслу запрос ближе к «a», по словам совпадает только «b»
    hits = chunks_repo.search_hybrid(conn, "KTL-1700", 5, 0.5, [doc_id], rrf_k=60)
    assert {h["chunk_id"] for h in hits} == {ids["a"], ids["b"]}  # «c» отсёк порог
    both = chunks_repo.search_hybrid(conn, "мотор 800", 5, 0.5, [doc_id], rrf_k=60)
    assert both[0]["chunk_id"] == ids["a"]
    assert both[0]["score"] == pytest.approx(2 / 61)  # найден обоими поисками


@pytest.mark.db
def test_filters(conn, corpus):
    doc_id, ids, _q = corpus
    hits = chunks_repo.search_vector(conn, "x", 5, 0.0, [doc_id], articul="KTL-1700")
    assert [h["chunk_id"] for h in hits] == [ids["b"]]
    assert chunks_repo.search_vector(conn, "x", 5, 0.0, ["other"]) == []


@pytest.mark.db
def test_suspicious_chunks_excluded_and_missing_embeddings(conn, corpus):
    from app.repositories import documents as documents_repo

    doc_id, ids, _q = corpus
    documents_repo.mark_chunks_suspicious(conn, [ids["a"]])
    hits = chunks_repo.search_vector(conn, "x", 5, 0.0, [doc_id])
    assert ids["a"] not in [h["chunk_id"] for h in hits]
    assert chunks_repo.missing_embeddings(conn, 10, doc_id=doc_id) == []
    conn.execute("UPDATE chunks SET embedding = NULL WHERE doc_id = %s", (doc_id,))
    assert len(chunks_repo.missing_embeddings(conn, 10, doc_id=doc_id)) == 3
    assert set(chunks_repo.get_chunks_by_ids(conn, [ids["b"], "nope"])) == {ids["b"]}
