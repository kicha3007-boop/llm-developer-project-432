"""Репозиторий фрагментов: запись, векторы и три режима поиска.

- ``vector_search`` — ближайшие по косинусному расстоянию (индекс HNSW, vector_cosine_ops);
- ``text_search`` — полнотекстовый поиск с ранжированием ts_rank_cd (индекс GIN);
- ``hybrid_search`` — слияние двух списков по обратным позициям (Reciprocal Rank Fusion):
  фрагмент получает 1 / (k + позиция) от каждого списка, где встретился. Складываются позиции,
  а не оценки: косинусная близость и ts_rank несопоставимы по шкале.

Порог релевантности — косинусная близость; он отсекает векторную ветку (и в гибриде тоже).
Полнотекстовая ветка отсекается самим условием совпадения слов.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from psycopg.types.json import Jsonb

from app.core.db import connection

_HIT_COLUMNS = "c.id, c.document_id, c.text, c.page, c.section, c.metadata, c.kind, c.ord"

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


@dataclass(frozen=True)
class SearchFilters:
    document_ids: Sequence[str] | None = None
    articul: str | None = None
    brand: str | None = None


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    text: str
    score: float
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "text"
    similarity: float | None = None


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


def query_tokens(query: str) -> list[str]:
    tokens = []
    for tok in _TOKEN_RE.findall(query.lower()):
        tok = tok.strip(".,")
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        if tok not in tokens:
            tokens.append(tok)
    return tokens


def _filters_sql(filters: SearchFilters | None, params: dict[str, Any]) -> str:
    clauses = ["TRUE"]
    if filters is None:
        return "TRUE"
    if filters.document_ids:
        clauses.append("c.document_id = ANY(%(f_docs)s)")
        params["f_docs"] = list(filters.document_ids)
    if filters.articul:
        clauses.append(
            "(c.metadata ->> 'articul' = %(f_art)s OR c.metadata -> 'articuls' ? %(f_art)s)"
        )
        params["f_art"] = filters.articul
    if filters.brand:
        clauses.append("lower(c.metadata ->> 'brand') = lower(%(f_brand)s)")
        params["f_brand"] = filters.brand
    return " AND ".join(clauses)


def _tsquery_sql(tokens: list[str], params: dict[str, Any]) -> str:
    """OR из plainto_tsquery по каждому слову: «мотор 800 Вт» находит и фрагмент только с «800»."""
    parts = []
    for i, tok in enumerate(tokens):
        params[f"t{i}"] = tok
        parts.append(f"plainto_tsquery('simple', %(t{i})s)")
    return " || ".join(parts)


def _hit(row: dict[str, Any], score: float, similarity: float | None = None) -> Hit:
    return Hit(
        chunk_id=row["id"],
        document_id=row["document_id"],
        text=row["text"],
        score=float(score),
        page=row["page"],
        section=row["section"],
        metadata=row["metadata"] or {},
        kind=row["kind"],
        similarity=None if similarity is None else float(similarity),
    )


# --- Запись ------------------------------------------------------------------------------------


def replace_for_document(document_id: str, chunks: Sequence[Any]) -> int:
    """Перезаписывает фрагменты документа в одной транзакции (повторный разбор не плодит дубли).

    ``chunks`` — объекты с полями id, ord, kind, text, page, section, metadata.
    """
    with connection() as conn, conn.transaction():
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, document_id, ord, kind, text, page, section, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (c.id, document_id, c.ord, c.kind, c.text, c.page, c.section, Jsonb(c.metadata))
                    for c in chunks
                ],
            )
    return len(chunks)


def counts(document_id: str) -> tuple[int, int]:
    """(всего фрагментов, с вектором)."""
    with connection() as conn:
        row = conn.execute(
            """
            SELECT count(*) AS total, count(embedding) AS embedded
            FROM chunks WHERE document_id = %s
            """,
            (document_id,),
        ).fetchone()
    return (row["total"], row["embedded"]) if row else (0, 0)


def missing_embeddings(limit: int, document_id: str | None = None) -> list[dict[str, Any]]:
    """Фрагменты без вектора — для досчёта. Повторный запуск после досчёта вернёт пусто."""
    params: dict[str, Any] = {"limit": limit}
    where = "c.embedding IS NULL"
    if document_id:
        where += " AND c.document_id = %(doc)s"
        params["doc"] = document_id
    with connection() as conn:
        return conn.execute(
            f"""
            SELECT c.id, c.text, c.section, d.title
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE {where}
            ORDER BY c.document_id, c.ord
            LIMIT %(limit)s
            """,
            params,
        ).fetchall()


def save_embeddings(pairs: Sequence[tuple[str, Sequence[float]]], model: str) -> int:
    """Пишет векторы пачкой одним executemany."""
    if not pairs:
        return 0
    with connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
            [(vector_literal(vec), model, chunk_id) for chunk_id, vec in pairs],
        )
    return len(pairs)


# --- Чтение ------------------------------------------------------------------------------------


def existing_ids(chunk_ids: Sequence[str]) -> set[str]:
    if not chunk_ids:
        return set()
    with connection() as conn:
        rows = conn.execute(
            "SELECT id FROM chunks WHERE id = ANY(%s)", (list(chunk_ids),)
        ).fetchall()
    return {r["id"] for r in rows}


def get_many(chunk_ids: Sequence[str]) -> list[Hit]:
    if not chunk_ids:
        return []
    with connection() as conn:
        rows = conn.execute(
            f"SELECT {_HIT_COLUMNS} FROM chunks c WHERE c.id = ANY(%s) "
            "ORDER BY c.document_id, c.ord",
            (list(chunk_ids),),
        ).fetchall()
    return [_hit(r, 0.0) for r in rows]


def for_documents(document_ids: Sequence[str], limit: int = 200) -> list[Hit]:
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT {_HIT_COLUMNS} FROM chunks c
            WHERE c.document_id = ANY(%s) ORDER BY c.document_id, c.ord LIMIT %s
            """,
            (list(document_ids), limit),
        ).fetchall()
    return [_hit(r, 0.0) for r in rows]


# --- Поиск -------------------------------------------------------------------------------------


def vector_search(
    query_vector: Sequence[float],
    *,
    top_k: int,
    threshold: float = 0.0,
    filters: SearchFilters | None = None,
) -> list[Hit]:
    params: dict[str, Any] = {"q": vector_literal(query_vector), "k": top_k, "thr": threshold}
    where = _filters_sql(filters, params)
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM (
                SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> %(q)s::vector) AS similarity
                FROM chunks c
                WHERE c.embedding IS NOT NULL AND {where}
                ORDER BY c.embedding <=> %(q)s::vector
                LIMIT %(k)s
            ) s
            WHERE s.similarity >= %(thr)s
            ORDER BY s.similarity DESC
            """,
            params,
        ).fetchall()
    return [_hit(r, r["similarity"], r["similarity"]) for r in rows]


def text_search(query: str, *, top_k: int, filters: SearchFilters | None = None) -> list[Hit]:
    tokens = query_tokens(query)
    if not tokens:
        return []
    params: dict[str, Any] = {"k": top_k}
    tsq = _tsquery_sql(tokens, params)
    where = _filters_sql(filters, params)
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT {_HIT_COLUMNS}, ts_rank_cd(c.tsv, q.query) AS rank
            FROM chunks c, (SELECT {tsq} AS query) q
            WHERE c.tsv @@ q.query AND {where}
            ORDER BY rank DESC, c.id
            LIMIT %(k)s
            """,
            params,
        ).fetchall()
    return [_hit(r, r["rank"]) for r in rows]


def hybrid_search(
    query: str,
    query_vector: Sequence[float],
    *,
    top_k: int,
    threshold: float = 0.0,
    rrf_k: int = 60,
    candidates: int | None = None,
    filters: SearchFilters | None = None,
) -> list[Hit]:
    """Гибрид: RRF по позициям в векторном и полнотекстовом списках, одной SQL-командой."""
    cand = candidates or max(top_k * 4, 20)
    params: dict[str, Any] = {
        "q": vector_literal(query_vector),
        "cand": cand,
        "thr": threshold,
        "rrf_k": rrf_k,
        "k": top_k,
    }
    where = _filters_sql(filters, params)
    tokens = query_tokens(query)
    if tokens:
        tsq = _tsquery_sql(tokens, params)
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
    vf AS (SELECT id, r, similarity FROM v WHERE similarity >= %(thr)s),
    {text_cte},
    fused AS (
        SELECT id, sum(1.0 / (%(rrf_k)s + r)) AS score
        FROM (SELECT id, r FROM vf UNION ALL SELECT id, r FROM t) u
        GROUP BY id
    )
    SELECT {_HIT_COLUMNS}, f.score, vf.similarity
    FROM fused f
    JOIN chunks c ON c.id = f.id
    LEFT JOIN vf ON vf.id = f.id
    ORDER BY f.score DESC, c.id
    LIMIT %(k)s
    """
    with connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_hit(r, r["score"], r["similarity"]) for r in rows]
