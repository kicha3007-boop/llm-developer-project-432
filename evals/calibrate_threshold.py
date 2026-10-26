"""Калибровка порога релевантности на своих документах.

    python -m evals.calibrate_threshold [--model intfloat/multilingual-e5-base] [--no-prefix]

Без базы и без сервера модели: документы из ``data/`` разбираются и режутся тем же кодом, что и в
сервисе, фрагменты кодируются локальной моделью эмбеддингов, затем для каждого контрольного
запроса печатается близость к верному фрагменту и к лучшему неверному. Порог выбирается между
распределениями «верных» и «мусора»; ``--no-prefix`` показывает, как без префиксов модели
просаживается близость релевантных пар.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from app.parsers import ParseError, parse_any
from app.parsers.chunker import chunk_blocks
from app.parsers.normalizer import guess_title, normalize
from app.rag.embedder import SentenceTransformerEmbedder, document_input

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["blender_passport.pdf", "blender_kp.docx", "kettle_spec.xlsx", "kettle_manual.pdf"]

# (запрос, подстрока, по которой опознаётся верный фрагмент)
PROBES = [
    ("Какая мощность у блендера", "Мощность: 800 Вт"),
    ("мотор 800 Вт", "Мощность: 800 Вт"),
    ("сколько скоростей у блендера МиксерПро", "Количество скоростей: 6"),
    ("гарантийный срок блендера", "Гарантия 24 месяца с даты продажи"),
    ("что входит в комплект блендера", "Венчик для взбивания"),
    ("можно ли мыть насадку в посудомойке", "посудомоечной машине"),
    ("объём чайника КеттлПро", "Объём: 1,7 л"),
    ("из чего сделан корпус чайника", "Материал корпуса"),
    ("KTL-1700", "KTL-1700: Бренд"),
    ("цена блендера BLD-800 оптом", "BLD-800: Модель"),
    ("условия доставки", "Доставка по РФ"),
    ("фильтр от накипи", "фильтр от накипи"),
]
# Запросы, на которые в документах ответа нет: их лучшая близость — уровень «мусора»
NEGATIVE = [
    "погода в Москве на выходные",
    "рецепт борща",
    "ремонт карбюратора автомобиля",
    "курс доллара на сегодня",
    "стиральная машина с сушкой",
    "размер экрана смартфона",
]


def load_chunks() -> list[tuple[str, str, str | None, str | None]]:
    chunks = []
    for name in DOCS:
        path = ROOT / "data" / name
        try:
            blocks = normalize(parse_any(path, path.suffix))
        except ParseError:
            continue
        title = guess_title(blocks)
        for c in chunk_blocks(blocks, name, name):
            chunks.append((c["text"], name, c["metadata"].get("section"), title))
    return chunks


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-prefix", action="store_true")
    args = parser.parse_args(argv)
    kwargs = {"model_name": args.model} if args.model else {}
    if args.no_prefix:
        kwargs.update(query_prefix="", document_prefix="")
    embedder = SentenceTransformerEmbedder(**kwargs)
    chunks = load_chunks()
    vectors = embedder.embed_documents([document_input(t, s, title) for t, _n, s, title in chunks])
    relevant, best_wrong = [], []
    print(
        f"модель: {embedder.model_name}, префиксы: {not args.no_prefix}, фрагментов: {len(chunks)}"
    )
    print(f"{'запрос':<42}{'верный':>8}{'ранг':>6}{'лучший неверный':>17}")
    for query, marker in PROBES:
        qv = embedder.embed_query(query)
        scored = sorted(
            ((dot(qv, v), text) for v, (text, *_rest) in zip(vectors, chunks, strict=True)),
            reverse=True,
        )
        good = [(i, s) for i, (s, text) in enumerate(scored) if marker in text]
        rank, good_sim = good[0] if good else (None, 0.0)
        wrong = max((s for s, text in scored if marker not in text), default=0.0)
        relevant.append(good_sim)
        best_wrong.append(wrong)
        print(f"{query[:40]:<42}{good_sim:>8.3f}{(rank or 0) + 1:>6}{wrong:>17.3f}")
    noise = []
    for query in NEGATIVE:
        qv = embedder.embed_query(query)
        noise.append(max(dot(qv, v) for v in vectors))
        print(f"{query[:40]:<42}{'—':>8}{'—':>6}{noise[-1]:>17.3f}")
    print(
        f"\nверные: min={min(relevant):.3f} median={statistics.median(relevant):.3f}; "
        f"лучшие неверные: median={statistics.median(best_wrong):.3f} max={max(best_wrong):.3f}; "
        f"запросы без ответа: median={statistics.median(noise):.3f} max={max(noise):.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
