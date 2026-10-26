"""Репозиторий фрагментов: запись, векторы и три режима поиска. Соединение передаёт вызывающий.

- ``search_vector`` — ближайшие по косинусному расстоянию (индекс HNSW, vector_cosine_ops);
- ``search_fts`` — полнотекстовый поиск с ранжированием ts_rank_cd (индекс GIN);
- ``search_hybrid`` — слияние двух списков по обратным позициям (Reciprocal Rank Fusion):
  фрагмент получает 1 / (k + позиция) от каждого списка, где встретился. Складываются позиции,
  а не оценки: косинусная близость и ts_rank несопоставимы по шкале.

Порог релевантности — косинусная близость; он отсекает векторную ветку (и в гибриде тоже).
Полнотекстовая ветка отсекается самим условием совпадения слов. Результат поиска — словари
``{"chunk_id", "doc_id", "text", "metadata", "page", "section", "filename", "score", ...}``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import settings
from app.rag.embedder import embed_query

_HIT_COLUMNS = (
    "c.id AS chunk_id, c.doc_id, c.text, c.metadata, c.page, c.section, c.kind, c.ordinal, "
    "c.suspicious, d.filename"
)
_FROM = "chunks c JOIN documents d ON d.id = c.doc_id"

# Вопросительные и служебные слова, которые с конфигурацией 'simple' дают только шум
_STOPWORDS = frozenset(
    [
        "и",
        "в",
        "во",
        "на",
        "по",
        "с",
        "со",
        "у",
        "к",
        "о",
        "об",
        "от",
        "до",
        "из",
        "за",
        "для",
        "что",
        "как",
        "какой",
        "какая",
        "какое",
        "какие",
        "какого",
        "это",
        "ли",
        "же",
        "а",
        "но",
        "или",
        "не",
        "ни",
        "есть",
        "the",
        "a",
        "an",
        "of",
        "is",
        "what",
        "which",
    ]
)
_TOKEN_RE = re.compile(r"[\w][\w\-.,]*[\w]|\w", re.UNICODE)


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


def query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(query.lower()):
        tok = tok.strip(".,")
        if len(tok) >= 2 and tok not in _STOPWORDS and tok not in tokens:
            tokens.append(tok)
    return tokens


def _filters(
    params: dict[str, Any],
    doc_ids: Sequence[str] | None,
    articul: str | None,
    brand: str | None,
) -> str:
    clauses = ["NOT c.suspicious"]
    if doc_ids is not None:
        clauses.append("c.doc_id = ANY(%(f_docs)s)")
        params["f_docs"] = list(doc_ids)
    if articul:
        clauses.append(
            "(c.metadata ->> 'articul' = %(f_art)s OR c.metadata -> 'articuls' ? %(f_art)s)"
        )
        params["f_art"] = articul
    if brand:
        clauses.append("lower(c.metadata ->> 'brand') = lower(%(f_brand)s)")
        params["f_brand"] = brand
    return " AND ".join(clauses)


def _tsquery(tokens: list[str], params: dict[str, Any]) -> str:
    """OR из plainto_tsquery по каждому слову: «мотор 800 Вт» находит и фрагмент только с «800»."""
    parts = []
    for i, tok in enumerate(tokens):
        params[f"t{i}"] = tok
        parts.append(f"plainto_tsquery('simple', %(t{i})s)")
    return " || ".join(parts)


def _hits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        meta = row.get("metadata") or {}
        row["metadata"] = meta
        if row.get("page") is None:
            row["page"] = meta.get("page")
        if row.get("section") is None:
            row["section"] = meta.get("section")
        row["score"] = float(row["score"])
        if row.get("similarity") is not None:
            row["similarity"] = float(row["similarity"])
    return rows


# --- Запись ------------------------------------------------------------------------------------


def replace_chunks(conn: psycopg.Connection, doc_id: str, chunks: Sequence[dict[str, Any]]) -> int:
    """Перезаписывает фрагменты документа одной транзакцией (повторный разбор без дублей).

    ``chunks`` — словари резки: ``{"id", "ordinal", "kind", "text", "metadata"}``.
    """
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            """
            INSERT INTO chunks (id, doc_id, ordinal, kind, text, page, section, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    c["id"],
                    doc_id,
                    c["ordinal"],
                    c.get("kind", "text"),
                    c["text"],
                    c["metadata"].get("page"),
                    c["metadata"].get("section"),
                    Jsonb(c["metadata"]),
                )
                for c in chunks
            ],
        )
    return len(chunks)


def save_embeddings(
    conn: psycopg.Connection, pairs: Sequence[tuple[str, Sequence[float]]], model: str
) -> int:
    """Пишет векторы пачкой одним executemany."""
    if not pairs:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
            [(vector_literal(vec), model, chunk_id) for chunk_id, vec in pairs],
        )
    return len(pairs)


# --- Чтение ------------------------------------------------------------------------------------


def list_chunks(conn: psycopg.Connection, doc_id: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, doc_id, ordinal, kind, text, page, section, metadata, suspicious "
            "FROM chunks WHERE doc_id = %s ORDER BY ordinal",
            (doc_id,),
        )
        return cur.fetchall()


def count_chunks(conn: psycopg.Connection, doc_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def get_chunks_by_ids(
    conn: psycopg.Connection, chunk_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Фрагменты по идентификаторам: ``{chunk_id: строка}``. Отсутствующих в базе нет в ответе."""
    if not chunk_ids:
        return {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {_HIT_COLUMNS} FROM {_FROM} WHERE c.id = ANY(%s)", (list(chunk_ids),))
        rows = cur.fetchall()
    for row in rows:
        meta = row.get("metadata") or {}
        row["metadata"] = meta
        row["page"] = row["page"] if row["page"] is not None else meta.get("page")
    return {row["chunk_id"]: row for row in rows}


def missing_embeddings(
    conn: psycopg.Connection, limit: int, doc_id: str | None = None
) -> list[dict[str, Any]]:
    """Фрагменты без вектора — для досчёта. Повторный запуск после досчёта вернёт пусто."""
    params: dict[str, Any] = {"limit": limit}
    where = "c.embedding IS NULL"
    if doc_id:
        where += " AND c.doc_id = %(doc)s"
        params["doc"] = doc_id
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT c.id, c.text, c.section, d.title FROM {_FROM}
            WHERE {where} ORDER BY c.doc_id, c.ordinal LIMIT %(limit)s
            """,
            params,
        )
        return cur.fetchall()


# --- Поиск -------------------------------------------------------------------------------------


def search_vector(
    conn: psycopg.Connection,
    query: str,
    top_k: int = 8,
    threshold: float | None = None,
    doc_ids: Sequence[str] | None = None,
    *,
    articul: str | None = None,
    brand: str | None = None,
    vector: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    thr = settings.relevance_threshold if threshold is None else threshold
    params: dict[str, Any] = {
        "q": vector_literal(vector if vector is not None else embed_query(query)),
        "k": top_k,
        "thr": thr,
    }
    where = _filters(params, doc_ids, articul, brand)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT * FROM (
                SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> %(q)s::vector) AS similarity
                FROM {_FROM}
                WHERE c.embedding IS NOT NULL AND {where}
                ORDER BY c.embedding <=> %(q)s::vector
                LIMIT %(k)s
            ) s
            WHERE s.similarity >= %(thr)s - 1e-6
            ORDER BY s.similarity DESC
            """,
            params,
        )
        rows = cur.fetchall()
    for row in rows:
        row["score"] = row["similarity"]
    return _hits(rows)


def search_fts(
    conn: psycopg.Connection,
    query: str,
    top_k: int = 8,
    doc_ids: Sequence[str] | None = None,
    *,
    articul: str | None = None,
    brand: str | None = None,
) -> list[dict[str, Any]]:
    tokens = query_tokens(query)
    if not tokens:
        return []
    params: dict[str, Any] = {"k": top_k}
    tsq = _tsquery(tokens, params)
    where = _filters(params, doc_ids, articul, brand)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT {_HIT_COLUMNS}, ts_rank_cd(c.tsv, q.query) AS score
            FROM {_FROM}, (SELECT {tsq} AS query) q
            WHERE c.tsv @@ q.query AND {where}
            ORDER BY score DESC, c.id
            LIMIT %(k)s
            """,
            params,
        )
        return _hits(cur.fetchall())


def search_hybrid(
    conn: psycopg.Connection,
    query: str,
    top_k: int = 8,
    threshold: float | None = None,
    doc_ids: Sequence[str] | None = None,
    *,
    articul: str | None = None,
    brand: str | None = None,
    vector: Sequence[float] | None = None,
    rrf_k: int | None = None,
    candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Гибрид: RRF по позициям в векторном и полнотекстовом списках, одной SQL-командой."""
    thr = settings.relevance_threshold if threshold is None else threshold
    params: dict[str, Any] = {
        "q": vector_literal(vector if vector is not None else embed_query(query)),
        "cand": candidates or max(top_k * 4, 20),
        "thr": thr,
        "rrf_k": rrf_k or settings.rrf_k,
        "k": top_k,
    }
    where = _filters(params, doc_ids, articul, brand)
    tokens = query_tokens(query)
    if tokens:
        tsq = _tsquery(tokens, params)
        text_cte = f"""
        t AS (
            SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q.query) DESC, c.id) AS r
            FROM chunks c, (SELECT {tsq} AS query) q
            WHERE c.tsv @@ q.query AND {where}
            ORDER BY ts_rank_cd(c.tsv, q.query) DESC, c.id
            LIMIT %(cand)s
        )"""
    else:
        text_cte = "t AS (SELECT NULL::text AS id, NULL::bigint AS r WHERE FALSE)"
    sql = f"""
    WITH v AS (
        SELECT c.id,
               row_number() OVER (ORDER BY c.embedding <=> %(q)s::vector) AS r,
               1 - (c.embedding <=> %(q)s::vector) AS similarity
        FROM chunks c
        WHERE c.embedding IS NOT NULL AND {where}
        ORDER BY c.embedding <=> %(q)s::vector
        LIMIT %(cand)s
    ),
    vf AS (SELECT id, r, similarity FROM v WHERE similarity >= %(thr)s - 1e-6),
    {text_cte},
    fused AS (
        SELECT id, sum(1.0 / (%(rrf_k)s + r)) AS score
        FROM (SELECT id, r FROM vf UNION ALL SELECT id, r FROM t) u
        GROUP BY id
    )
    SELECT {_HIT_COLUMNS}, f.score, vf.similarity
    FROM fused f
    JOIN chunks c ON c.id = f.id
    JOIN documents d ON d.id = c.doc_id
    LEFT JOIN vf ON vf.id = f.id
    ORDER BY f.score DESC, c.id
    LIMIT %(k)s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return _hits(cur.fetchall())
