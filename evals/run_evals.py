"""Прогон метрик генерации на эталоне.

    python -m evals.run_evals           # подмножество eval_defaults (быстро на локальной модели)
    python -m evals.run_evals --all     # все документы эталона
    make evals / make evals ALL=1

Нужны поднятая инфраструктура, применённые миграции, сервер модели и модель эмбеддингов.
Печатает таблицу по документам и средние, называет самую слабую метрику и сохраняет отчёт
в ``evals/reports/report-<время>.json`` и ``.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.core.db import close_pool, open_pool
from app.core.logging import bind_context, clear_context, get_logger, setup_logging
from app.guardrails.pii import PiiMasker
from app.rag.context import ContextItem, format_context
from app.repositories import chunks as chunks_repo
from app.repositories import llm_calls
from app.services import documents as documents_service
from app.services.rag_pipeline import run_rag_pipeline
from evals.metrics import characteristics_score, citation_score, judge_card, weakest_metric

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden_cards.json"
REPORTS = ROOT / "evals" / "reports"
METRICS = ("characteristics", "citation", "faithfulness")


async def evaluate_document(name: str, golden: dict[str, Any], run_id: str) -> dict[str, Any]:
    path = ROOT / "data" / name
    upload = documents_service.register_upload(name, path.read_bytes())
    doc_id = upload.document["id"]
    processed = documents_service.process_document(doc_id)
    row: dict[str, Any] = {"document": name, "document_id": doc_id}
    if processed["status"] == "failed":
        row.update(status="failed", reason=processed.get("error"))
        return row

    job_id = f"eval-{run_id}-{doc_id}"
    clear_context()
    bind_context(job_id=job_id)
    started = time.perf_counter()
    result = await run_rag_pipeline([doc_id], golden.get("product"))
    row["seconds"] = round(time.perf_counter() - started, 1)
    row["status"] = result.status
    row["attempts"] = result.attempts
    row["review_reason"] = result.review_reason
    card = result.card.model_dump() if result.card else {}
    row["card"] = card

    hits = chunks_repo.get_many(result.context_chunk_ids)
    chunks = {h.chunk_id: {"text": h.text, "page": h.page} for h in hits}
    expected = golden.get("characteristics") or {}
    chars = characteristics_score(expected, card) if expected else None
    cites = citation_score(card, chunks, golden.get("source_probes"))
    row["characteristics"] = round(chars.score, 3) if chars else None
    row["characteristics_detail"] = [vars(d) for d in chars.details if not d.ok] if chars else []
    row["citation"] = round(cites.score, 3)
    row["citation_probes"] = round(cites.probe_score, 3)
    row["citation_problems"] = cites.problems

    masker = PiiMasker()
    items = [
        ContextItem(
            h.chunk_id,
            h.document_id,
            h.metadata.get("filename"),
            h.page,
            h.section,
            masker.mask(h.text),
        )
        for h in hits
    ]
    context_text, _ = format_context(items, 100_000)
    if card:
        verdict = await judge_card(card, context_text)
        row["faithfulness"] = round(verdict.score, 3)
        row["unsupported_claims"] = verdict.unsupported_claims
    else:
        row["faithfulness"] = None

    missing_expected = golden.get("missing_fields") or []
    row["missing_expected"] = missing_expected
    row["missing_actual"] = card.get("missing_fields", [])
    row["confidence"] = card.get("confidence")
    must_not = (golden.get("injection_must_be_blocked") or {}).get("card_must_not_contain", [])
    flat = json.dumps(card, ensure_ascii=False)
    row["injection_leaks"] = [s for s in must_not if s in flat]
    cost = llm_calls.cost_for_job(job_id)
    row["cost"] = str(cost.get("cost", Decimal(0)))
    row["calls"] = cost.get("calls", 0)
    return row


def averages(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for metric in METRICS:
        values = [r[metric] for r in rows if isinstance(r.get(metric), (int, float))]
        if values:
            result[metric] = round(sum(values) / len(values), 3)
    return result


def render_table(rows: list[dict[str, Any]], avg: dict[str, float]) -> str:
    header = f"{'документ':<24}{'статус':<14}{'характ.':>9}{'цитир.':>9}{'подтв.':>9}{'стоим.':>12}"
    lines = [header, "-" * len(header)]

    def fmt(v: Any) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    for r in rows:
        lines.append(
            f"{r['document']:<24}{r.get('status', ''):<14}{fmt(r.get('characteristics')):>9}"
            f"{fmt(r.get('citation')):>9}{fmt(r.get('faithfulness')):>9}{r.get('cost', '—'):>12}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'среднее':<38}{fmt(avg.get('characteristics')):>9}{fmt(avg.get('citation')):>9}"
        f"{fmt(avg.get('faithfulness')):>9}"
    )
    return "\n".join(lines)


async def run(all_docs: bool, only: list[str] | None) -> dict[str, Any]:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    names = only or (list(golden["documents"]) if all_docs else golden["eval_defaults"])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    rows = []
    for name in names:
        log.info("eval_document", document=name)
        rows.append(await evaluate_document(name, golden["documents"][name], run_id))
    avg = averages(rows)
    weakest = weakest_metric(avg) if avg else None
    return {"run_id": run_id, "documents": names, "rows": rows, "averages": avg, "weakest": weakest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогон метрик генерации на эталоне")
    parser.add_argument("--all", action="store_true", help="весь набор документов эталона")
    parser.add_argument("--doc", action="append", help="конкретный документ из эталона")
    args = parser.parse_args(argv)
    setup_logging()
    open_pool(wait=True)
    try:
        report = asyncio.run(run(args.all, args.doc))
    finally:
        close_pool()
    table = render_table(report["rows"], report["averages"])
    print(table)
    if report["weakest"]:
        print(f"\nСамая слабая метрика: {report['weakest'][0]} = {report['weakest'][1]:.2f}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = REPORTS / f"report-{report['run_id']}"
    base.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    base.with_suffix(".md").write_text(
        f"# Прогон метрик {report['run_id']}\n\n```text\n{table}\n```\n", encoding="utf-8"
    )
    print(f"Отчёт сохранён: {base.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
