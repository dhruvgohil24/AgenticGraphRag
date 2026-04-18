# run_eval.py
"""
Evaluation Suite Entrypoint — v2.0

Usage:
    python run_eval.py                          # run full question bank
    python run_eval.py --limit 5                # run first 5 questions only
    python run_eval.py --category factual       # run one category only
    python run_eval.py --id q001 q003 q005      # run specific question IDs

Output:
    data/eval/results/eval_YYYYMMDD_HHMMSS.csv
    data/eval/results/eval_YYYYMMDD_HHMMSS.json
    Terminal summary table
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from tqdm import tqdm

from src.agent.workflow import AgentResponse, AgentWorkflow
from src.eval.evaluator import (
    EvalQuestion,
    EvalResult,
    evaluate_response,
    load_question_bank,
)
from src.eval.report import print_summary, save_csv, save_json
from src.utils import get_logger, load_config

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup() -> tuple[AgentWorkflow, dict]:
    """Load config, configure LlamaIndex Settings, instantiate workflow."""
    load_dotenv()
    config = load_config()

    Settings.llm = Ollama(
        model=config["ollama"]["llm_model"],
        base_url=config["ollama"]["base_url"],
        request_timeout=config["ollama"]["request_timeout"],
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config["ollama"]["embed_model"],
        base_url=config["ollama"]["base_url"],
    )

    workflow = AgentWorkflow(config)
    logger.info("✅ AgentWorkflow initialized for evaluation.")
    return workflow, config


# ─────────────────────────────────────────────────────────────────────────────
# Async Runner — same safe pattern as app.py
# ─────────────────────────────────────────────────────────────────────────────

def _run_workflow_sync(
    workflow: AgentWorkflow,
    query: str,
) -> AgentResponse:
    """
    Run the async AgentWorkflow from a synchronous context.
    Uses a fresh event loop in a dedicated thread — safe for sequential calls.
    """
    def _thread_target() -> AgentResponse:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(workflow.run(query))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_thread_target)
        # Per-question timeout: 10 minutes max
        return future.result(timeout=600)


# ─────────────────────────────────────────────────────────────────────────────
# Core Eval Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    questions: list[EvalQuestion],
    workflow: AgentWorkflow,
) -> list[EvalResult]:
    """
    Sequential evaluation loop with progress bar.

    For each question:
        1. Run AgentWorkflow to get a response
        2. Run evaluate_response to score all four metrics
        3. Append EvalResult to results list

    Sequential execution avoids Neo4j connection pool exhaustion.
    Each question takes ~2-4 minutes on M-series Mac with local Ollama.

    Args:
        questions: Filtered list of EvalQuestion objects.
        workflow:  Initialized AgentWorkflow instance.

    Returns:
        List of EvalResult objects, one per question.
    """
    results: list[EvalResult] = []
    total = len(questions)
    start_time = time.time()

    print(f"\n🚀 Starting evaluation — {total} question(s)")
    print(f"   Estimated time: {total * 3}-{total * 5} minutes on M-series Mac")
    print(f"   Each question: ~3-5 min (synthesis + 3 self-correction + 4 eval metrics)\n")

    for idx, q in enumerate(
        tqdm(questions, desc="Questions", unit="q"), start=1
    ):
        q_start = time.time()
        print(f"\n{'━'*60}")
        print(f"[{idx}/{total}] {q.id} | {q.category}")
        print(f"Q: {q.question[:80]}...")
        print("━" * 60)

        # ── Step A: Run the agent ─────────────────────────────────────────────
        try:
            print("   🤖 Running AgentWorkflow...")
            agent_response = _run_workflow_sync(workflow, q.question)
            print(
                f"   ✅ Agent done — "
                f"verdict={agent_response.verdict} | "
                f"score={agent_response.composite_score:.2f}/10 | "
                f"{agent_response.latency_seconds}s"
            )
        except Exception as e:
            import traceback
            error_str = traceback.format_exc()
            logger.error(f"AgentWorkflow failed for [{q.id}]: {e}")
            # Create a failed result rather than crashing the whole eval
            results.append(EvalResult(
                question_id=q.id,
                category=q.category,
                question=q.question,
                ground_truth=q.ground_truth,
                error=error_str,
            ))
            print(f"   ❌ Agent failed — skipping metric scoring.")
            continue

        # ── Step B: Score all metrics ─────────────────────────────────────────
        try:
            print("   📐 Running metric evaluation...")
            eval_result = evaluate_response(q, agent_response)
        except Exception as e:
            import traceback
            error_str = traceback.format_exc()
            logger.error(f"Metric evaluation failed for [{q.id}]: {e}")
            eval_result = EvalResult(
                question_id=q.id,
                category=q.category,
                question=q.question,
                ground_truth=q.ground_truth,
                final_answer=agent_response.final_answer,
                verdict=agent_response.verdict,
                agent_composite_score=agent_response.composite_score,
                latency_seconds=agent_response.latency_seconds,
                error=error_str,
            )

        results.append(eval_result)

        q_elapsed = round(time.time() - q_start, 1)
        remaining = total - idx
        avg_so_far = (time.time() - start_time) / idx
        eta_minutes = round(remaining * avg_so_far / 60, 1)

        print(
            f"\n   📊 [{q.id}] Done in {q_elapsed}s | "
            f"Overall={eval_result.overall_eval_score:.4f} | "
            f"{'✅ PASS' if eval_result.passed else '❌ FAIL'}"
        )
        print(f"   ⏳ ETA for remaining {remaining} question(s): ~{eta_minutes} min")

    total_elapsed = round((time.time() - start_time) / 60, 1)
    print(f"\n✅ Evaluation complete in {total_elapsed} minutes.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic GraphRAG v2.0 — Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py                     # full question bank
  python run_eval.py --limit 3           # first 3 questions (quick smoke test)
  python run_eval.py --category factual  # only factual questions
  python run_eval.py --id q001 q003      # specific question IDs
  python run_eval.py --no-json           # CSV only, skip JSON output
        """,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to one category: factual | relational | conceptual",
    )
    parser.add_argument(
        "--id",
        nargs="+",
        default=None,
        help="Run specific question IDs only (e.g. --id q001 q003)",
    )
    parser.add_argument(
        "--bank",
        type=str,
        default="data/eval/question_bank.json",
        help="Path to question bank JSON (default: data/eval/question_bank.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval/results",
        help="Output directory for CSV/JSON (default: data/eval/results)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON output, write CSV only",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("═" * 65)
    print("   AGENTIC GRAPHRAG v2.0 — EVALUATION SUITE")
    print("═" * 65)

    # ── Load question bank ────────────────────────────────────────────────────
    if not Path(args.bank).exists():
        print(f"❌ Question bank not found at: {args.bank}")
        print("   Create it at data/eval/question_bank.json")
        sys.exit(1)

    questions = load_question_bank(args.bank)

    # ── Apply filters ─────────────────────────────────────────────────────────
    if args.id:
        questions = [q for q in questions if q.id in args.id]
        print(f"🔍 Filtered to IDs {args.id} → {len(questions)} question(s)")

    if args.category:
        questions = [q for q in questions if q.category == args.category]
        print(
            f"🔍 Filtered to category '{args.category}' "
            f"→ {len(questions)} question(s)"
        )

    if args.limit:
        questions = questions[: args.limit]
        print(f"🔍 Limiting to first {args.limit} question(s)")

    if not questions:
        print("❌ No questions match the specified filters.")
        sys.exit(1)

    print(f"\n📋 Running {len(questions)} question(s):\n")
    for q in questions:
        print(f"   {q.id} [{q.category}] — {q.question[:65]}...")

    # ── Initialize workflow ───────────────────────────────────────────────────
    print("\n🔧 Initializing AgentWorkflow...")
    workflow, config = setup()

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = run_evaluation(questions, workflow)

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path = save_csv(results, output_dir=args.output_dir)

    if not args.no_json:
        json_path = save_json(results, output_dir=args.output_dir)
    else:
        json_path = None

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(results)

    print("📁 Output files:")
    print(f"   CSV  : {csv_path}")
    if json_path:
        print(f"   JSON : {json_path}")
    print()


if __name__ == "__main__":
    main()