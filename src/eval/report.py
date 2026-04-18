# src/eval/report.py
"""
Evaluation Report Generator — v2.0

Takes a list of EvalResult objects and produces:
  1. A timestamped CSV file in data/eval/results/
  2. A human-readable summary printed to terminal
  3. A per-category breakdown table
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.eval.evaluator import EvalResult
from src.utils import get_logger

logger = get_logger(__name__)

# CSV column order — matches EvalResult fields
CSV_COLUMNS = [
    "question_id",
    "category",
    "question",
    "ground_truth",
    "final_answer",
    "verdict",
    "agent_composite_score",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "overall_eval_score",
    "passed",
    "latency_seconds",
    "attempt_count",
    "was_corrected",
    "vector_nodes_retrieved",
    "graph_nodes_retrieved",
    "error",
]


def _safe_mean(values: list[float]) -> float:
    """Mean of a list, ignoring zeros from errored results."""
    valid = [v for v in values if v > 0.0]
    return round(float(np.mean(valid)), 4) if valid else 0.0


def save_csv(
    results: list[EvalResult],
    output_dir: str = "data/eval/results",
) -> str:
    """
    Save all EvalResult objects to a timestamped CSV.

    Args:
        results:    List of completed EvalResult objects.
        output_dir: Directory to write the CSV into.

    Returns:
        Absolute path to the written CSV file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{timestamp}.csv"
    filepath = Path(output_dir) / filename

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for result in results:
            row = asdict(result)
            # Truncate long text fields for CSV readability
            row["question"]      = row["question"][:200]
            row["ground_truth"]  = row["ground_truth"][:300]
            row["final_answer"]  = row["final_answer"][:300]
            writer.writerow(row)

    logger.info(f"💾 CSV saved to: {filepath}")
    return str(filepath)


def save_json(
    results: list[EvalResult],
    output_dir: str = "data/eval/results",
) -> str:
    """
    Save full results as JSON (includes complete answers, no truncation).
    Useful for deeper analysis in a notebook.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(output_dir) / f"eval_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"💾 JSON saved to: {filepath}")
    return str(filepath)


def _category_breakdown(results: list[EvalResult]) -> dict[str, dict]:
    """Group results by category and compute per-category averages."""
    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    breakdown = {}
    for cat, cat_results in categories.items():
        breakdown[cat] = {
            "count": len(cat_results),
            "pass_rate": round(
                sum(r.passed for r in cat_results) / len(cat_results), 3
            ),
            "faithfulness":      _safe_mean([r.faithfulness for r in cat_results]),
            "answer_relevancy":  _safe_mean([r.answer_relevancy for r in cat_results]),
            "context_precision": _safe_mean([r.context_precision for r in cat_results]),
            "context_recall":    _safe_mean([r.context_recall for r in cat_results]),
            "overall":           _safe_mean([r.overall_eval_score for r in cat_results]),
            "avg_latency":       _safe_mean([r.latency_seconds for r in cat_results]),
        }
    return breakdown


def print_summary(results: list[EvalResult]) -> None:
    """
    Print a comprehensive human-readable summary to the terminal.
    Designed to look impressive in a demo or presentation recording.
    """
    total = len(results)
    errored = [r for r in results if r.error]
    valid = [r for r in results if not r.error]
    passed = [r for r in valid if r.passed]

    # ── Overall Banner ────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("   AGENTIC GRAPHRAG v2.0 — EVALUATION REPORT")
    print("═" * 65)
    print(f"   Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total Qs   : {total}")
    print(f"   Evaluated  : {len(valid)} | Errored: {len(errored)}")
    print(f"   Pass Rate  : {len(passed)}/{len(valid)} "
          f"({100*len(passed)/max(len(valid),1):.1f}%) "
          f"[threshold ≥ 0.65]")
    print("─" * 65)

    # ── Aggregate Metrics ─────────────────────────────────────────────────────
    if valid:
        avg_faith   = _safe_mean([r.faithfulness for r in valid])
        avg_rel     = _safe_mean([r.answer_relevancy for r in valid])
        avg_prec    = _safe_mean([r.context_precision for r in valid])
        avg_recall  = _safe_mean([r.context_recall for r in valid])
        avg_overall = _safe_mean([r.overall_eval_score for r in valid])
        avg_latency = _safe_mean([r.latency_seconds for r in valid])
        avg_agent   = _safe_mean([r.agent_composite_score for r in valid])
        pct_corrected = 100 * sum(r.was_corrected for r in valid) / len(valid)
        avg_attempts  = _safe_mean([float(r.attempt_count) for r in valid])

        print(f"\n  {'METRIC':<26} {'SCORE':>8}  {'VISUAL':}")
        print("  " + "─" * 55)

        metrics = [
            ("Faithfulness",      avg_faith,   "Ragas-compatible"),
            ("Answer Relevancy",  avg_rel,     "Embedding-based"),
            ("Context Precision", avg_prec,    "Rank-weighted"),
            ("Context Recall",    avg_recall,  "GT coverage"),
        ]
        for name, score, note in metrics:
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            color_indicator = (
                "✅" if score >= 0.75 else
                "⚠️ " if score >= 0.55 else
                "❌"
            )
            print(
                f"  {color_indicator} {name:<24} "
                f"{score:>6.4f}  [{bar}]  {note}"
            )

        print("  " + "─" * 55)
        bar_len = int(avg_overall * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  🏆 {'OVERALL (mean)':<24} {avg_overall:>6.4f}  [{bar}]")

        print(f"\n  Agent Self-Score (composite) : {avg_agent:.2f}/10")
        print(f"  Avg Latency per query        : {avg_latency:.1f}s")
        print(f"  % Answers Corrected          : {pct_corrected:.1f}%")
        print(f"  Avg Attempts per query       : {avg_attempts:.2f}")

    # ── Category Breakdown ────────────────────────────────────────────────────
    if valid:
        breakdown = _category_breakdown(valid)
        print("\n" + "─" * 65)
        print("  CATEGORY BREAKDOWN")
        print("  " + "─" * 63)
        header = (
            f"  {'Category':<16} {'N':>3}  {'Pass%':>6}  "
            f"{'Faith':>6}  {'Relev':>6}  "
            f"{'Prec':>6}  {'Recall':>6}  {'Overall':>7}"
        )
        print(header)
        print("  " + "─" * 63)
        for cat, stats in sorted(breakdown.items()):
            print(
                f"  {cat:<16} {stats['count']:>3}  "
                f"{stats['pass_rate']*100:>5.1f}%  "
                f"{stats['faithfulness']:>6.4f}  "
                f"{stats['answer_relevancy']:>6.4f}  "
                f"{stats['context_precision']:>6.4f}  "
                f"{stats['context_recall']:>6.4f}  "
                f"{stats['overall']:>7.4f}"
            )

    # ── Per-Question Results ──────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  PER-QUESTION RESULTS")
    print("  " + "─" * 63)
    print(
        f"  {'ID':<6}  {'Cat':<12}  {'Overall':>7}  "
        f"{'Verdict':<14}  {'Lat':>6}  Status"
    )
    print("  " + "─" * 63)

    for r in results:
        if r.error:
            status = "❌ ERROR"
            overall_str = "  N/A "
            verdict_str = "ERROR"
        else:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            overall_str = f"{r.overall_eval_score:.4f}"
            verdict_str = r.verdict[:14]

        print(
            f"  {r.question_id:<6}  {r.category:<12}  {overall_str:>7}  "
            f"{verdict_str:<14}  {r.latency_seconds:>5.1f}s  {status}"
        )

    # ── Slowest Queries ───────────────────────────────────────────────────────
    if valid:
        slowest = sorted(valid, key=lambda r: r.latency_seconds, reverse=True)[:3]
        print("\n" + "─" * 65)
        print("  SLOWEST QUERIES")
        for r in slowest:
            print(
                f"  {r.question_id} | {r.latency_seconds:.1f}s | "
                f"attempts={r.attempt_count} | "
                f"'{r.question[:55]}...'"
            )

    print("\n" + "═" * 65 + "\n")