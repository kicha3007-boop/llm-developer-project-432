"""Поиск по фрагментам: по смыслу, по словам и гибрид (RRF), с фильтрами и порогом.

Команда: ``python -m app.rag.retrieval "мотор 800 Вт" --mode hybrid [--doc ID] [--articul X]``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.core.config import settings
from app.rag.embedder import get_embedder
from app.repositories import chunks as chunks_repo
from app.repositories.chunks import Hit, SearchFilters

MODES = ("vector", "text", "hybrid")


def search(
    query: str,
    *,
    mode: str = "hybrid",
    top_k: int | None = None,
    document_ids: Sequence[str] | None = None,
    articul: str | None = None,
    brand: str | None = None,
    threshold: float | None = None,
) -> list[Hit]:
    if mode not in MODES:
        raise ValueError(f"неизвестный режим поиска: {mode}")
    top_k = top_k or settings.search_top_k
    thr = settings.relevance_threshold if threshold is None else threshold
    filters = SearchFilters(document_ids=document_ids, articul=articul, brand=brand)
    if mode == "text":
        return chunks_repo.text_search(query, top_k=top_k, filters=filters)
    vector = get_embedder().embed_query(query)
    if mode == "vector":
        return chunks_repo.vector_search(vector, top_k=top_k, threshold=thr, filters=filters)
    return chunks_repo.hybrid_search(
        query, vector, top_k=top_k, threshold=thr, rrf_k=settings.rrf_k, filters=filters
    )


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Эталонная формула RRF на Python: Σ 1 / (k + позиция). В проде её считает SQL
    (``chunks.hybrid_search``); здесь — для калибровки и тестов на заданных списках."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Поиск по фрагментам документов")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--doc", action="append", dest="docs")
    parser.add_argument("--articul")
    parser.add_argument("--brand")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args(argv)
    hits = search(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        document_ids=args.docs,
        articul=args.articul,
        brand=args.brand,
        threshold=args.threshold,
    )
    if not hits:
        print("Ничего не найдено (порог релевантности отсёк всё или совпадений нет)")
    for hit in hits:
        sim = f" sim={hit.similarity:.3f}" if hit.similarity is not None else ""
        print(
            f"[{hit.chunk_id}] score={hit.score:.4f}{sim} doc={hit.document_id} "
            f"p={hit.page} § {hit.section}"
        )
        print("   " + hit.text.replace("\n", " | ")[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
