# src/agent/self_correct.py
"""
DSPy-Inspired Self-Correction Module — v2.0

Scores answers on three independent metrics before returning to the user.
Architecture is inspired by DSPy's "Signature + Metric" pattern:
each metric is a narrow, focused LLM call with a strict output schema.

Metrics:
┌─────────────────────┬────────┬─────────────────────────────────────────┐
│ Metric              │ Weight │ What it measures                        │
├─────────────────────┼────────┼─────────────────────────────────────────┤
│ Groundedness        │  0.50  │ Every claim traceable to context        │
│ Context Relevance   │  0.30  │ Retrieved context addresses the query   │
│ Answer Completeness │  0.20  │ Answer fully resolves what was asked    │
└─────────────────────┴────────┴─────────────────────────────────────────┘

Composite score = weighted average.
Threshold = 6.5 (configurable).
On failure: returns a ReQuerySignal with a targeted re-query string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import Settings

from src.utils import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CORRECTION_THRESHOLD: float = 6.5

METRIC_WEIGHTS: dict[str, float] = {
    "groundedness": 0.50,
    "context_relevance": 0.30,
    "answer_completeness": 0.20,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricScore:
    """Result of scoring one metric."""
    name: str
    score: float                     # 1.0 – 10.0
    reasoning: str                   # One-sentence explanation from LLM
    passed: bool = False             # score >= CORRECTION_THRESHOLD

    def __post_init__(self) -> None:
        self.passed = self.score >= CORRECTION_THRESHOLD


@dataclass
class CorrectionResult:
    """
    Full output of the self-correction module.
    Consumed by workflow.py and the Streamlit UI.
    """
    original_answer: str
    final_answer: str
    verdict: str                              # "APPROVED" | "CORRECTED" | "INSUFFICIENT_CONTEXT"
    composite_score: float
    metric_scores: dict[str, MetricScore] = field(default_factory=dict)
    was_corrected: bool = False
    requery_signal: Optional[str] = None     # targeted re-query if score < threshold
    correction_log: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Metric Prompt Templates
# Each prompt asks for EXACTLY two lines: SCORE and REASON.
# Strict format enforced to make parsing reliable.
# ─────────────────────────────────────────────────────────────────────────────

_GROUNDEDNESS_PROMPT = """\
You are a strict factual auditor. Your only job is to check whether \
an ANSWER is supported by the CONTEXT.

Groundedness means: every factual claim in the answer can be directly \
traced to a specific statement in the context. Extrapolations, \
inferences not in the context, and outside knowledge all reduce the score.

CONTEXT:
{context}

ANSWER:
{answer}

Score groundedness from 1 to 10 where:
10 = every single claim is explicitly in the context
 7 = most claims supported, minor extrapolation present
 4 = significant claims not supported by context
 1 = answer contradicts or ignores the context entirely

Respond in EXACTLY this format — two lines, nothing else:
SCORE: <integer 1-10>
REASON: <one sentence explaining the score>
"""

_CONTEXT_RELEVANCE_PROMPT = """\
You are evaluating whether a retrieved CONTEXT is useful for answering a QUERY.

Context Relevance means: the retrieved passages actually contain information \
that addresses what the student is asking. Tangentially related or \
off-topic passages reduce the score.

QUERY:
{query}

CONTEXT:
{context}

Score context relevance from 1 to 10 where:
10 = context directly and completely addresses the query
 7 = context mostly relevant, some off-topic material
 4 = context partially relevant but misses key aspects of the query
 1 = context is entirely unrelated to the query

Respond in EXACTLY this format — two lines, nothing else:
SCORE: <integer 1-10>
REASON: <one sentence explaining the score>
"""

_COMPLETENESS_PROMPT = """\
You are evaluating whether an ANSWER fully resolves a student's QUESTION.

Answer Completeness means: the answer addresses all parts of the question \
without leaving major aspects unanswered. A complete answer may still be \
brief if the question is simple.

QUESTION:
{query}

ANSWER:
{answer}

Score completeness from 1 to 10 where:
10 = answer fully and clearly resolves every aspect of the question
 7 = answer addresses the main point but misses secondary aspects
 4 = answer partially addresses the question, significant gaps remain
 1 = answer does not address the question at all

Respond in EXACTLY this format — two lines, nothing else:
SCORE: <integer 1-10>
REASON: <one sentence explaining the score>
"""

_REQUERY_PROMPT = """\
An AI system retrieved context and generated an answer to a student question, \
but the answer failed quality checks on the following metrics:
{failed_metrics}

Original question: {query}
Failed answer: {answer}

Your job: write a BETTER search query that would retrieve more relevant, \
specific context to answer the original question correctly.

Rules:
- Make the query more specific than the original question
- Focus on the aspect that failed (e.g. if groundedness failed, \
  ask for more specific source material)
- The query must be a single sentence, no longer than 20 words
- Output ONLY the improved query, nothing else
"""

_REVISION_PROMPT = """\
You are a careful educational assistant. A previous answer failed quality \
checks. Rewrite it using ONLY the information in the CONTEXT below.

Rules:
- Use **bold** for key terms and concepts
- Structure the answer clearly — use numbered steps or bullet points \
  if the answer has multiple parts
- If the context does not fully support an answer, state exactly what \
  is and is not supported
- Do NOT add any information not present in the context

CONTEXT:
{context}

ORIGINAL QUESTION: {query}

FAILED ANSWER (do not copy this — rewrite from context only):
{answer}

QUALITY ISSUES FOUND:
{issues_summary}

REVISED ANSWER:
"""


# ─────────────────────────────────────────────────────────────────────────────
# Metric Scorer
# ─────────────────────────────────────────────────────────────────────────────

def _parse_score_response(raw: str, metric_name: str) -> tuple[float, str]:
    """
    Parse the strict two-line LLM response format.

    Expected:
        SCORE: 8
        REASON: The answer directly references the treaty mentioned in chunk 2.

    Returns (score, reason). Falls back to score=7.0 on parse failure
    so a single malformed response doesn't tank the whole pipeline.
    """
    score = 7.0    # fail-open default
    reason = "Could not parse LLM response."

    for line in raw.strip().splitlines():
        line = line.strip()
        upper = line.upper()

        if upper.startswith("SCORE:"):
            score_part = line.split(":", 1)[1].strip()
            nums = re.findall(r"\d+(?:\.\d+)?", score_part)
            if nums:
                score = max(1.0, min(10.0, float(nums[0])))

        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    logger.debug(f"   {metric_name}: {score}/10 — {reason}")
    return score, reason


def _score_metric(
    prompt: str,
    metric_name: str,
) -> MetricScore:
    """
    Run one focused LLM scoring call.
    Isolated into its own function for clean error handling per metric.
    """
    try:
        response = Settings.llm.complete(prompt)
        score, reason = _parse_score_response(response.text, metric_name)
        return MetricScore(name=metric_name, score=score, reasoning=reason)
    except Exception as e:
        logger.error(f"Metric '{metric_name}' scoring failed: {e} — defaulting to 7.0")
        return MetricScore(
            name=metric_name,
            score=7.0,
            reasoning=f"Scoring failed ({e}) — assigned neutral score.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Re-Query Signal Generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_requery(
    query: str,
    answer: str,
    failed_metrics: list[MetricScore],
) -> str:
    """
    Ask the LLM to produce a better, more targeted search query
    based on which specific metrics failed and why.
    """
    failed_summary = "\n".join(
        f"- {m.name} (score {m.score}/10): {m.reasoning}"
        for m in failed_metrics
    )

    prompt = _REQUERY_PROMPT.format(
        failed_metrics=failed_summary,
        query=query,
        answer=answer,
    )

    try:
        response = Settings.llm.complete(prompt)
        requery = response.text.strip()
        # Strip any accidental quotes the LLM might wrap the query in
        requery = requery.strip('"\'')
        logger.info(f"🔄 Re-query generated: '{requery}'")
        return requery
    except Exception as e:
        logger.error(f"Re-query generation failed: {e} — using original query.")
        return query


# ─────────────────────────────────────────────────────────────────────────────
# Answer Revisor
# ─────────────────────────────────────────────────────────────────────────────

def _revise_answer(
    query: str,
    answer: str,
    context: str,
    failed_metrics: list[MetricScore],
) -> str:
    """
    Rewrite the answer using only retrieved context.
    Called when composite score < threshold AND re-query also didn't help
    (i.e., this is the last-resort correction path).
    """
    issues_summary = "\n".join(
        f"- {m.name}: {m.reasoning}" for m in failed_metrics
    )

    prompt = _REVISION_PROMPT.format(
        context=context[:3000],
        query=query,
        answer=answer,
        issues_summary=issues_summary,
    )

    try:
        response = Settings.llm.complete(prompt)
        revised = response.text.strip()
        logger.info("✅ Answer revised by correction module.")
        return revised
    except Exception as e:
        logger.error(f"Answer revision failed: {e} — returning original.")
        return answer


# ─────────────────────────────────────────────────────────────────────────────
# Composite Scorer
# ─────────────────────────────────────────────────────────────────────────────

def _compute_composite(metric_scores: dict[str, MetricScore]) -> float:
    """
    Weighted average of metric scores.
    Uses METRIC_WEIGHTS dict — weights sum to 1.0.
    """
    total = sum(
        METRIC_WEIGHTS.get(name, 0.0) * ms.score
        for name, ms in metric_scores.items()
    )
    return round(total, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_self_correction(
    query: str,
    answer: str,
    context: str,
    attempt: int = 1,
) -> CorrectionResult:
    """
    Full self-correction pipeline for one answer.

    Runs three independent metric evaluations, computes a composite score,
    and either approves or triggers a correction signal.

    Args:
        query:   Original user question.
        answer:  Answer produced by the synthesis step.
        context: Fused context string from RRF retrieval.
        attempt: Which attempt this is (1 = first, 2 = after re-query).
                 On attempt 2 we revise in-place rather than re-querying.

    Returns:
        CorrectionResult with verdict, scores, and final answer.
    """
    result = CorrectionResult(
        original_answer=answer,
        final_answer=answer,
        verdict="APPROVED",
        composite_score=0.0,
    )

    result.correction_log.append(
        f"🔎 Self-Correction (attempt {attempt}/2) — "
        f"scoring on 3 metrics..."
    )

    # ── Handle no-context edge case ───────────────────────────────────────────
    if not context or len(context.strip()) < 20:
        result.verdict = "INSUFFICIENT_CONTEXT"
        result.composite_score = 0.0
        result.correction_log.append(
            "   ⚠️  Context too short to evaluate — "
            "answer may be unreliable."
        )
        result.correction_log.append("🏁 Verdict: INSUFFICIENT_CONTEXT")
        logger.warning("Self-correction: insufficient context.")
        return result

    # ── Score all three metrics ───────────────────────────────────────────────
    context_truncated = context[:2500]   # stay within LLM context window

    groundedness = _score_metric(
        _GROUNDEDNESS_PROMPT.format(
            context=context_truncated, answer=answer
        ),
        "groundedness",
    )

    context_relevance = _score_metric(
        _CONTEXT_RELEVANCE_PROMPT.format(
            query=query, context=context_truncated
        ),
        "context_relevance",
    )

    completeness = _score_metric(
        _COMPLETENESS_PROMPT.format(query=query, answer=answer),
        "answer_completeness",
    )

    metric_scores = {
        "groundedness": groundedness,
        "context_relevance": context_relevance,
        "answer_completeness": completeness,
    }
    result.metric_scores = metric_scores

    # ── Compute composite score ───────────────────────────────────────────────
    composite = _compute_composite(metric_scores)
    result.composite_score = composite

    result.correction_log.append(
        f"   └─ Groundedness   : {groundedness.score}/10 "
        f"(w=0.50) — {groundedness.reasoning}"
    )
    result.correction_log.append(
        f"   └─ Ctx Relevance  : {context_relevance.score}/10 "
        f"(w=0.30) — {context_relevance.reasoning}"
    )
    result.correction_log.append(
        f"   └─ Completeness   : {completeness.score}/10 "
        f"(w=0.20) — {completeness.reasoning}"
    )
    result.correction_log.append(
        f"   └─ Composite score: {composite:.2f} / 10.0 "
        f"(threshold={CORRECTION_THRESHOLD})"
    )

    # ── Decision ──────────────────────────────────────────────────────────────
    failed = [m for m in metric_scores.values() if not m.passed]

    if composite >= CORRECTION_THRESHOLD:
        result.verdict = "APPROVED"
        result.was_corrected = False
        result.correction_log.append(
            f"✅ Verdict: APPROVED "
            f"(composite {composite:.2f} ≥ {CORRECTION_THRESHOLD})"
        )

    elif attempt == 1:
        # First failure — generate a targeted re-query signal.
        # The workflow will use this to re-retrieve before revising.
        result.verdict = "REQUERY"
        result.was_corrected = True
        requery = _generate_requery(query, answer, failed)
        result.requery_signal = requery
        result.correction_log.append(
            f"⚠️  Verdict: REQUERY — composite {composite:.2f} "
            f"< {CORRECTION_THRESHOLD}"
        )
        result.correction_log.append(
            f"   └─ Targeted re-query: \"{requery}\""
        )

    else:
        # Second failure — revise in-place, no more re-queries
        result.verdict = "CORRECTED"
        result.was_corrected = True
        revised = _revise_answer(query, answer, context, failed)
        result.final_answer = revised
        result.correction_log.append(
            f"⚠️  Verdict: CORRECTED — composite {composite:.2f} "
            f"< {CORRECTION_THRESHOLD} on attempt 2"
        )
        result.correction_log.append(
            "   └─ Answer rewritten from context by Revisor."
        )

    result.correction_log.append(
        f"🏁 Final verdict: {result.verdict} | "
        f"Score: {composite:.2f}/10"
    )

    return result