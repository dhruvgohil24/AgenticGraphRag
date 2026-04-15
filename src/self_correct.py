# src/self_correct.py

"""
Self-Correction Loop — Critique-and-Revise Pattern

After the router retrieves an answer, this module:
  1. Scores the answer's faithfulness to retrieved context (1-10)
  2. If score < 7: rewrites the answer using ONLY the context
  3. Returns a CorrectedResponse with full audit trail for the UI
"""

from dataclasses import dataclass, field
from llama_index.core import Settings
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorrectedResponse:
    """
    Final output object handed to the Streamlit UI.

    Fields:
      original_answer   — what the router first produced
      final_answer      — approved or rewritten answer
      verdict           — "APPROVED" | "CORRECTED" | "INSUFFICIENT_CONTEXT"
      faithfulness_score — int 1-10 from the critic
      correction_log    — list of strings shown in the sidebar
      was_corrected     — bool flag for UI badge colouring
    """
    original_answer: str = ""
    final_answer: str = ""
    verdict: str = "APPROVED"
    faithfulness_score: int = 0
    correction_log: list[str] = field(default_factory=list)
    was_corrected: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Critic: Score faithfulness
# ─────────────────────────────────────────────────────────────────────────────

CRITIC_PROMPT = """\
You are a strict factual auditor for an educational AI system.

Your job is to check whether an ANSWER is fully supported by the CONTEXT provided.
You must give a FAITHFULNESS SCORE from 1 to 10:

  10 — Every single claim in the answer is directly supported by the context.
   7 — Most claims are supported; minor extrapolations present.
   4 — Some claims are supported but significant unsupported claims exist.
   1 — The answer contradicts or completely ignores the context.

After the score, write ISSUES: and list any specific claims in the answer
that are NOT supported by the context. If none, write ISSUES: None.

CONTEXT:
{context}

ANSWER:
{answer}

Respond in EXACTLY this format — nothing else:
SCORE: <number>
ISSUES: <list of issues or None>
"""


def critique_answer(answer: str, context: str) -> tuple[int, str]:
    """
    Ask the LLM to score the answer's faithfulness to context.
    Returns (score: int, issues: str).
    Falls back to score=10 if parsing fails (fail-open).
    """
    if not context or context.strip() == "":
        logger.warning("Empty context passed to critic — skipping critique.")
        return 10, "None"

    prompt = CRITIC_PROMPT.format(context=context[:3000], answer=answer)

    try:
        response = Settings.llm.complete(prompt)
        raw = response.text.strip()

        # Parse SCORE line
        score = 10  # default fail-open
        issues = "None"

        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    # Extract first integer found in case LLM adds words
                    import re
                    nums = re.findall(r'\d+', score_str)
                    if nums:
                        score = max(1, min(10, int(nums[0])))
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("ISSUES:"):
                issues = line.split(":", 1)[1].strip()

        logger.info(f"🔍 Critic score: {score}/10 | Issues: {issues}")
        return score, issues

    except Exception as e:
        logger.error(f"Critic LLM call failed: {e} — defaulting to score=10")
        return 10, "None"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Revisor: Rewrite the answer if score is too low
# ─────────────────────────────────────────────────────────────────────────────

REVISOR_PROMPT = """\
You are a careful educational assistant. A previous answer was found to contain \
claims not supported by the source material.

Your task: Rewrite the answer using ONLY information explicitly present in the \
CONTEXT below. Do NOT add any outside knowledge.
If the context does not contain enough information to answer the question fully, \
say: "Based on the available course material, I can only confirm that: ..." \
and state what IS supported.

ORIGINAL QUESTION: {query}

CONTEXT:
{context}

FLAWED ANSWER (do NOT copy this — rewrite from context only):
{answer}

ISSUES FOUND IN FLAWED ANSWER:
{issues}

CORRECTED ANSWER:
"""


def revise_answer(
    query: str,
    answer: str,
    context: str,
    issues: str,
) -> str:
    """
    Rewrite the answer to be strictly grounded in retrieved context.
    Returns the revised answer string.
    """
    prompt = REVISOR_PROMPT.format(
        query=query,
        context=context[:3000],
        answer=answer,
        issues=issues,
    )

    try:
        response = Settings.llm.complete(prompt)
        revised = response.text.strip()
        logger.info("✅ Answer successfully revised by Revisor.")
        return revised

    except Exception as e:
        logger.error(f"Revisor LLM call failed: {e} — returning original answer.")
        return answer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Master correction function (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

# Threshold below which the answer gets rewritten
FAITHFULNESS_THRESHOLD = 7


def self_correct(
    query: str,
    answer: str,
    retrieved_context: str,
    route_taken: str,
) -> CorrectedResponse:
    """
    Main entry point for the self-correction loop.
    Called after route_query() returns, before showing answer to user.

    Args:
        query            — original user question
        answer           — answer produced by the router
        retrieved_context — raw chunks / graph nodes used to generate answer
        route_taken      — "vector" or "graph" (for logging)

    Returns:
        CorrectedResponse with final_answer and full audit trail
    """
    result = CorrectedResponse(original_answer=answer)
    result.correction_log.append("🔎 Step 4: Running Self-Correction audit...")
    result.correction_log.append(
        f"   └─ Route was: {route_taken.upper()} | "
        f"Context length: {len(retrieved_context)} chars"
    )

    # ── Handle edge case: no context retrieved ────────────────────────────────
    if not retrieved_context or "No relevant context" in retrieved_context:
        result.final_answer = answer
        result.verdict = "INSUFFICIENT_CONTEXT"
        result.faithfulness_score = 0
        result.correction_log.append(
            "   └─ ⚠️  No context available — skipping critique. "
            "Answer may be unreliable."
        )
        result.correction_log.append(
            "🏁 Verdict: INSUFFICIENT_CONTEXT"
        )
        logger.warning("Self-correction skipped — no context available.")
        return result

    # ── Step 1: Critique ──────────────────────────────────────────────────────
    result.correction_log.append(
        "   └─ Critic LLM evaluating faithfulness..."
    )
    score, issues = critique_answer(answer, retrieved_context)
    result.faithfulness_score = score

    result.correction_log.append(
        f"   └─ Faithfulness Score: {score}/10"
    )
    result.correction_log.append(
        f"   └─ Issues found: {issues}"
    )

    # ── Step 2: Decide — approve or revise ───────────────────────────────────
    if score >= FAITHFULNESS_THRESHOLD:
        result.final_answer = answer
        result.verdict = "APPROVED"
        result.was_corrected = False
        result.correction_log.append(
            f"✅ Verdict: APPROVED (score {score} ≥ threshold {FAITHFULNESS_THRESHOLD})"
        )
        logger.info(f"Answer APPROVED — score {score}/10")

    else:
        result.correction_log.append(
            f"⚠️  Score {score} < threshold {FAITHFULNESS_THRESHOLD} "
            "— triggering Revisor..."
        )
        revised = revise_answer(query, answer, retrieved_context, issues)
        result.final_answer = revised
        result.verdict = "CORRECTED"
        result.was_corrected = True
        result.correction_log.append(
            "   └─ Revisor rewrote answer using only retrieved context."
        )
        result.correction_log.append(
            "🏁 Verdict: CORRECTED"
        )
        logger.info(f"Answer CORRECTED — score was {score}/10")

    return result