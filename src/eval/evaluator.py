# src/eval/evaluator.py
"""
Quantitative Evaluation Engine — v2.0

Implements four metrics with identical mathematical definitions to Ragas,
but using local Ollama instead of OpenAI — no API keys required.

Metrics:
┌──────────────────────┬──────────────────────────────────────────────────────┐
│ Metric               │ Definition                                           │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ Faithfulness         │ |supported claims| / |total claims in answer|        │
│ Answer Relevancy     │ mean cosine_sim(reverse_questions, original_question) │
│ Context Precision    │ Σ(precision@k * rel_k) / Σ(rel_k)  [weighted]       │
│ Context Recall       │ |GT claims covered by context| / |GT claims|         │
└──────────────────────┴──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from llama_index.core import Settings

from src.agent.workflow import AgentResponse
from src.retrieval.rrf_fusion import FusedResult
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalQuestion:
    """One entry from question_bank.json."""
    id: str
    category: str
    question: str
    ground_truth: str
    expected_route: str = "hybrid"


@dataclass
class EvalResult:
    """
    Complete evaluation result for one question.
    Every field maps directly to a CSV column.
    """
    # Question metadata
    question_id: str
    category: str
    question: str
    ground_truth: str

    # Agent output
    final_answer: str = ""
    verdict: str = ""
    agent_composite_score: float = 0.0
    latency_seconds: float = 0.0
    attempt_count: int = 1
    was_corrected: bool = False
    vector_nodes_retrieved: int = 0
    graph_nodes_retrieved: int = 0

    # Eval metrics (Ragas-compatible names)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Aggregate
    overall_eval_score: float = 0.0
    passed: bool = False

    # Error tracking
    error: Optional[str] = None

    def __post_init__(self) -> None:
        # Compute overall as equal-weighted mean of four metrics
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        valid = [s for s in scores if s > 0.0]
        self.overall_eval_score = round(
            float(np.mean(valid)) if valid else 0.0, 4
        )
        self.passed = self.overall_eval_score >= 0.65


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# Each prompt has a strict, parseable output schema.
# ─────────────────────────────────────────────────────────────────────────────

_CLAIM_EXTRACTION_PROMPT = """\
Extract all individual factual claims from the following text.
A claim is a single, atomic statement that can be independently verified.

Rules:
- One claim per line
- Each claim must be a complete sentence
- Do not include vague or opinion statements
- Maximum 15 claims

TEXT:
{text}

Output ONLY the numbered list of claims, nothing else:
1. <claim>
2. <claim>
...
"""

_CLAIM_VERIFICATION_PROMPT = """\
You are a strict factual verifier.
Determine whether the following CLAIM is directly supported by the CONTEXT.

CONTEXT:
{context}

CLAIM:
{claim}

Reply with ONLY one word:
SUPPORTED   ← if the context explicitly contains information that supports this claim
UNSUPPORTED ← if the context does not contain information to support this claim
"""

_REVERSE_QUESTION_PROMPT = """\
Given the following ANSWER, generate {n} different questions that this answer \
could be responding to.

Rules:
- Each question must be answerable using only the information in the answer
- Questions should be specific, not vague
- Output ONLY the questions, one per line, no numbering

ANSWER:
{answer}

{n} QUESTIONS:
"""

_CONTEXT_RELEVANCE_PROMPT = """\
Is the following CONTEXT CHUNK relevant for answering the QUESTION?

QUESTION: {question}

CONTEXT CHUNK:
{chunk}

Reply with ONLY one word:
RELEVANT
IRRELEVANT
"""


# ─────────────────────────────────────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _extract_claims(text: str) -> list[str]:
    """
    Use LLM to extract atomic factual claims from a text.
    Returns list of claim strings.
    Handles malformed LLM output gracefully.
    """
    if not text or len(text.strip()) < 20:
        return []

    prompt = _CLAIM_EXTRACTION_PROMPT.format(text=text[:2000])

    try:
        response = Settings.llm.complete(prompt)
        raw = response.text.strip()

        claims = []
        for line in raw.splitlines():
            line = line.strip()
            # Strip leading number and punctuation: "1. ", "2) ", etc.
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if len(cleaned) > 15:   # skip very short/empty lines
                claims.append(cleaned)

        logger.debug(f"   Extracted {len(claims)} claims from text.")
        return claims[:15]  # hard cap

    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        return []


def _verify_claim(claim: str, context: str) -> bool:
    """
    Ask LLM whether a single claim is supported by context.
    Returns True if supported, False otherwise.
    Fails open (returns True) on LLM errors to avoid false negatives.
    """
    prompt = _CLAIM_VERIFICATION_PROMPT.format(
        context=context[:2500],
        claim=claim,
    )
    try:
        response = Settings.llm.complete(prompt)
        raw = response.text.strip().upper()
        # Be robust — LLM might say "SUPPORTED because..." or "YES, SUPPORTED"
        return "SUPPORTED" in raw and "UNSUPPORTED" not in raw
    except Exception as e:
        logger.error(f"Claim verification failed: {e} — defaulting to supported")
        return True


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _is_chunk_relevant(question: str, chunk_text: str) -> bool:
    """
    Ask LLM if a retrieved chunk is relevant to the question.
    Returns True if relevant.
    """
    prompt = _CONTEXT_RELEVANCE_PROMPT.format(
        question=question,
        chunk=chunk_text[:800],
    )
    try:
        response = Settings.llm.complete(prompt)
        raw = response.text.strip().upper()
        return "RELEVANT" in raw and "IRRELEVANT" not in raw
    except Exception as e:
        logger.error(f"Context relevance check failed: {e}")
        return True  # fail open


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Faithfulness
# Definition: fraction of answer claims supported by retrieved context
# Range: [0.0, 1.0] — higher is better
# ─────────────────────────────────────────────────────────────────────────────

def score_faithfulness(
    answer: str,
    context: str,
) -> float:
    """
    Faithfulness = |supported_claims| / |total_claims|

    Algorithm:
        1. Extract atomic claims from the answer via LLM
        2. For each claim, verify against context via LLM
        3. Return fraction of supported claims

    Returns 0.0 if no claims could be extracted.
    Returns 1.0 if context is empty (cannot disprove anything).
    """
    if not context or len(context.strip()) < 20:
        logger.debug("   Faithfulness: no context — returning 1.0")
        return 1.0

    claims = _extract_claims(answer)
    if not claims:
        logger.debug("   Faithfulness: no claims extracted — returning 0.5")
        return 0.5

    supported = sum(
        1 for claim in claims
        if _verify_claim(claim, context)
    )

    score = round(supported / len(claims), 4)
    logger.debug(
        f"   Faithfulness: {supported}/{len(claims)} claims supported → {score}"
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — Answer Relevancy
# Definition: mean cosine similarity between reverse-generated questions
#             and the original question
# Range: [0.0, 1.0] — higher is better
# ─────────────────────────────────────────────────────────────────────────────

def score_answer_relevancy(
    question: str,
    answer: str,
    n_reverse: int = 3,
) -> float:
    """
    Answer Relevancy = mean cosine_sim(reverse_q_i, original_q)

    Algorithm:
        1. Ask LLM to generate n_reverse questions that the answer responds to
        2. Embed each reverse question and the original question
        3. Compute cosine similarity for each pair
        4. Return the mean

    This metric is embedding-based — avoids LLM-as-judge score inflation.
    """
    if not answer or len(answer.strip()) < 20:
        return 0.0

    # Generate reverse questions
    prompt = _REVERSE_QUESTION_PROMPT.format(answer=answer[:1500], n=n_reverse)
    try:
        response = Settings.llm.complete(prompt)
        raw = response.text.strip()
        reverse_questions = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 10
        ][:n_reverse]
    except Exception as e:
        logger.error(f"   Answer relevancy: reverse question generation failed: {e}")
        return 0.5

    if not reverse_questions:
        return 0.5

    # Embed original question
    try:
        orig_embedding = Settings.embed_model.get_text_embedding(question)
    except Exception as e:
        logger.error(f"   Answer relevancy: embedding failed: {e}")
        return 0.5

    # Compute mean cosine similarity
    similarities = []
    for rq in reverse_questions:
        try:
            rq_embedding = Settings.embed_model.get_text_embedding(rq)
            sim = _cosine_similarity(orig_embedding, rq_embedding)
            similarities.append(sim)
            logger.debug(f"   Reverse Q: '{rq[:60]}' | sim={sim:.4f}")
        except Exception as e:
            logger.error(f"   Reverse question embedding failed: {e}")

    if not similarities:
        return 0.5

    score = round(float(np.mean(similarities)), 4)
    logger.debug(f"   Answer Relevancy: mean sim={score}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — Context Precision
# Definition: weighted precision rewarding relevant chunks ranked higher
# Range: [0.0, 1.0] — higher is better
# ─────────────────────────────────────────────────────────────────────────────

def score_context_precision(
    question: str,
    retrieved_chunks: list[str],
) -> float:
    """
    Context Precision (weighted) = Σ(precision@k * rel_k) / Σ(rel_k)

    Where:
        rel_k       = 1 if chunk at rank k is relevant, 0 otherwise
        precision@k = fraction of top-k chunks that are relevant

    This metric rewards having relevant chunks at the TOP of the ranked list.
    A system that returns relevant chunks at ranks 1,2 scores higher than one
    that returns the same relevant chunks at ranks 4,5.

    Returns 0.0 if no chunks retrieved.
    Returns 1.0 if no chunks are relevant (edge case — avoid division by zero).
    """
    if not retrieved_chunks:
        return 0.0

    # Binary relevance per chunk
    relevance_flags: list[bool] = []
    for chunk in retrieved_chunks:
        is_relevant = _is_chunk_relevant(question, chunk)
        relevance_flags.append(is_relevant)
        logger.debug(
            f"   Chunk rank {len(relevance_flags)}: "
            f"{'RELEVANT' if is_relevant else 'IRRELEVANT'} | "
            f"'{chunk[:60]}...'"
        )

    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        logger.debug("   Context Precision: no relevant chunks → 0.0")
        return 0.0

    # Weighted precision computation
    weighted_sum = 0.0
    running_relevant = 0

    for k, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            running_relevant += 1
            precision_at_k = running_relevant / k
            weighted_sum += precision_at_k

    score = round(weighted_sum / total_relevant, 4)
    logger.debug(
        f"   Context Precision: {total_relevant}/{len(retrieved_chunks)} "
        f"relevant → weighted precision={score}"
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4 — Context Recall
# Definition: fraction of ground truth claims covered by retrieved context
# Range: [0.0, 1.0] — higher is better
# ─────────────────────────────────────────────────────────────────────────────

def score_context_recall(
    ground_truth: str,
    context: str,
) -> float:
    """
    Context Recall = |GT claims supported by context| / |GT claims|

    Algorithm:
        1. Extract atomic claims from the ground truth answer
        2. For each claim, check if any retrieved context chunk supports it
        3. Return fraction of covered ground truth claims

    This metric requires ground truth — it measures whether the retrieval
    system found the information needed to answer correctly.
    """
    if not ground_truth or len(ground_truth.strip()) < 20:
        logger.debug("   Context Recall: no ground truth → returning 0.5")
        return 0.5

    if not context or len(context.strip()) < 20:
        logger.debug("   Context Recall: no context → returning 0.0")
        return 0.0

    gt_claims = _extract_claims(ground_truth)
    if not gt_claims:
        logger.debug("   Context Recall: no GT claims extracted → returning 0.5")
        return 0.5

    covered = sum(
        1 for claim in gt_claims
        if _verify_claim(claim, context)
    )

    score = round(covered / len(gt_claims), 4)
    logger.debug(
        f"   Context Recall: {covered}/{len(gt_claims)} "
        f"GT claims covered → {score}"
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_response(
    eval_q: EvalQuestion,
    agent_response: AgentResponse,
) -> EvalResult:
    """
    Run all four metrics on one agent response.

    Args:
        eval_q:         The question + ground truth from question_bank.json
        agent_response: Full AgentResponse from AgentWorkflow.run()

    Returns:
        EvalResult with all metric scores populated.
    """
    logger.info(
        f"\n{'─'*50}\n"
        f"Evaluating [{eval_q.id}] — {eval_q.category}\n"
        f"Q: {eval_q.question[:80]}...\n"
        f"{'─'*50}"
    )

    # Extract retrieval counts from fused result
    vector_count = 0
    graph_count = 0
    retrieved_chunk_texts: list[str] = []
    context_text = ""

    if agent_response.fused_result:
        fr: FusedResult = agent_response.fused_result
        vector_count = len(fr.vector_nodes)
        graph_count = len(fr.graph_nodes)
        retrieved_chunk_texts = [n.text for n in fr.fused_nodes]
        context_text = fr.context_text

    result = EvalResult(
        question_id=eval_q.id,
        category=eval_q.category,
        question=eval_q.question,
        ground_truth=eval_q.ground_truth,
        final_answer=agent_response.final_answer,
        verdict=agent_response.verdict,
        agent_composite_score=agent_response.composite_score,
        latency_seconds=agent_response.latency_seconds,
        attempt_count=agent_response.attempt_count,
        was_corrected=agent_response.was_corrected,
        vector_nodes_retrieved=vector_count,
        graph_nodes_retrieved=graph_count,
        error=agent_response.error,
    )

    if agent_response.error:
        logger.warning(
            f"   ⚠️  Agent returned error for [{eval_q.id}] — "
            "skipping metric scoring."
        )
        return result

    # ── Metric 1: Faithfulness ────────────────────────────────────────────────
    logger.info("   📐 Scoring Faithfulness...")
    t0 = time.time()
    result.faithfulness = score_faithfulness(
        answer=agent_response.final_answer,
        context=context_text,
    )
    logger.info(
        f"   ✅ Faithfulness = {result.faithfulness:.4f} "
        f"({time.time()-t0:.1f}s)"
    )

    # ── Metric 2: Answer Relevancy ────────────────────────────────────────────
    logger.info("   📐 Scoring Answer Relevancy...")
    t0 = time.time()
    result.answer_relevancy = score_answer_relevancy(
        question=eval_q.question,
        answer=agent_response.final_answer,
        n_reverse=3,
    )
    logger.info(
        f"   ✅ Answer Relevancy = {result.answer_relevancy:.4f} "
        f"({time.time()-t0:.1f}s)"
    )

    # ── Metric 3: Context Precision ───────────────────────────────────────────
    logger.info("   📐 Scoring Context Precision...")
    t0 = time.time()
    result.context_precision = score_context_precision(
        question=eval_q.question,
        retrieved_chunks=retrieved_chunk_texts,
    )
    logger.info(
        f"   ✅ Context Precision = {result.context_precision:.4f} "
        f"({time.time()-t0:.1f}s)"
    )

    # ── Metric 4: Context Recall ──────────────────────────────────────────────
    logger.info("   📐 Scoring Context Recall...")
    t0 = time.time()
    result.context_recall = score_context_recall(
        ground_truth=eval_q.ground_truth,
        context=context_text,
    )
    logger.info(
        f"   ✅ Context Recall = {result.context_recall:.4f} "
        f"({time.time()-t0:.1f}s)"
    )

    # Recompute overall now that all metric fields are set
    scores = [
        result.faithfulness,
        result.answer_relevancy,
        result.context_precision,
        result.context_recall,
    ]
    result.overall_eval_score = round(float(np.mean(scores)), 4)
    result.passed = result.overall_eval_score >= 0.65

    logger.info(
        f"\n   📊 [{eval_q.id}] Summary:\n"
        f"      Faithfulness     : {result.faithfulness:.4f}\n"
        f"      Answer Relevancy : {result.answer_relevancy:.4f}\n"
        f"      Context Precision: {result.context_precision:.4f}\n"
        f"      Context Recall   : {result.context_recall:.4f}\n"
        f"      Overall          : {result.overall_eval_score:.4f} "
        f"({'PASS' if result.passed else 'FAIL'})"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Question Bank Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_question_bank(path: str = "data/eval/question_bank.json") -> list[EvalQuestion]:
    """
    Load and validate the question bank JSON file.
    Skips placeholder questions (those containing 'REPLACE THIS').
    """
    with open(path, "r", encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    questions = []
    skipped = 0
    for entry in raw:
        # Skip template placeholder questions
        if "REPLACE THIS" in entry.get("question", ""):
            skipped += 1
            continue
        questions.append(EvalQuestion(
            id=entry["id"],
            category=entry.get("category", "general"),
            question=entry["question"],
            ground_truth=entry.get("ground_truth", ""),
            expected_route=entry.get("expected_route", "hybrid"),
        ))

    logger.info(
        f"📋 Loaded {len(questions)} questions from question bank "
        f"({skipped} placeholder(s) skipped)."
    )
    return questions