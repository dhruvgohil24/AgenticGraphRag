# src/agent/workflow.py
"""
Agent Workflow Orchestrator — v2.0

Ties together RRF retrieval + synthesis + self-correction
into a single typed async pipeline.

Flow:
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Step 1: RETRIEVE │  ← parallel RRF (vector + graph → fused)
└──────┬───────────┘
       │  FusedResult
       ▼
┌──────────────────┐
│ Step 2: SYNTHESIZE│  ← LLM generates answer from fused context
└──────┬────────────┘
       │  (answer, context)
       ▼
┌──────────────────────┐
│ Step 3: SELF-CORRECT │  ← 3-metric scoring
└──────┬───────────────┘
       │
   ┌───┴──────────────────────────────┐
   │ APPROVED          REQUERY        │
   │    │                │            │
   │    ▼                ▼            │
   │  DONE       Step 4: RE-RETRIEVE  │
   │             (targeted re-query)  │
   │                     │            │
   │                     ▼            │
   │             Step 5: RE-SYNTHESIZE│
   │                     │            │
   │                     ▼            │
   │             Step 6: FINAL-CORRECT│
   │             (attempt 2 — revises │
   │              in-place if needed) │
   └──────────────────────────────────┘
       │
       ▼
┌──────────────────┐
│  AgentResponse   │  ← handed to Streamlit app.py
└──────────────────┘
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import Settings

from src.agent.self_correct import CorrectionResult, run_self_correction
from src.retrieval.rrf_fusion import FusedResult, fuse_retrieve
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Contract — AgentResponse
# This is the single object app.py receives and renders.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """
    Complete output from one query through the full pipeline.

    All fields are consumed by the Streamlit UI:
      - final_answer       → main chat bubble
      - thought_log        → sidebar step-by-step log
      - metric_scores      → sidebar score display
      - source_attributions → "Sources used:" section in chat
    """
    query: str
    final_answer: str = ""
    verdict: str = "APPROVED"
    composite_score: float = 0.0
    was_corrected: bool = False
    route_summary: str = "hybrid"
    latency_seconds: float = 0.0
    attempt_count: int = 1

    # Detailed logs for the sidebar
    thought_log: list[str] = field(default_factory=list)

    # Per-metric scores for the UI score card
    metric_scores: dict = field(default_factory=dict)

    # Source attribution: which lectures contributed to the answer
    source_attributions: list[str] = field(default_factory=list)

    # Raw retrieval result (for evaluation suite)
    fused_result: Optional[FusedResult] = None

    # Error state
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Synthesis
# Called after retrieval, produces the initial answer from fused context.
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
You are an expert educational assistant. Answer the student's question \
using ONLY the context provided below.

The context comes from a hybrid retrieval system that searched both a \
knowledge graph and a vector database — it represents the most relevant \
passages from the course materials.

Formatting rules:
- Use **bold** for key terms, theorems, and named concepts
- Use numbered lists for processes, steps, or sequences
- Use bullet points for properties, characteristics, or lists of items
- Keep mathematical terms precise — do not paraphrase formal definitions
- If the context contains a direct definition, quote it then explain it
- End with a one-sentence summary if the answer is longer than 3 paragraphs
- If the context is insufficient, state exactly what is missing rather \
  than guessing

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def _synthesize_answer(query: str, context: str) -> str:
    """
    Generate an answer from fused context using the LLM.
    Synchronous — called from within async steps via run_in_executor.
    """
    if not context or len(context.strip()) < 20:
        return (
            "I could not find sufficient context in the course materials "
            "to answer this question. Please check that relevant lecture "
            "notes have been ingested."
        )

    prompt = _SYNTHESIS_PROMPT.format(
        context=context[:3500],
        query=query,
    )

    try:
        response = Settings.llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Synthesis LLM call failed: {e}")
        return f"Synthesis failed: {e}"


def _extract_source_attributions(fused_result: FusedResult) -> list[str]:
    """
    Build a deduplicated list of source lectures from fused nodes.
    Used for the "Sources:" section in the chat UI.
    """
    seen: set[str] = set()
    sources: list[str] = []

    for node in fused_result.fused_nodes[:6]:
        lecture = node.source_lecture
        source_tag = f"{node.source.upper()}"
        entry = f"{lecture} [{source_tag}]"
        if entry not in seen:
            seen.add(entry)
            sources.append(entry)

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Steps
# Each step is a private async method.
# Public interface is the single `run()` coroutine.
# ─────────────────────────────────────────────────────────────────────────────

class AgentWorkflow:
    """
    Stateless async workflow orchestrator.
    Instantiated once at Streamlit startup and reused across queries.

    Usage:
        workflow = AgentWorkflow(config)
        response: AgentResponse = await workflow.run("What caused WW1?")
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────

    async def _step_retrieve(
        self,
        query: str,
        log: list[str],
        top_k: int = 6,
    ) -> FusedResult:
        """
        Parallel RRF retrieval. Queries both DBs concurrently and fuses results.
        """
        log.append("🔀 Step 1: Hybrid RRF Retrieval")
        log.append(
            f"   └─ Querying Vector DB (ChromaDB) + "
            f"Graph DB (Neo4j) in parallel..."
        )

        fused = await fuse_retrieve(query, self.config, top_k=top_k)

        log.extend(fused.retrieval_log)
        log.append(
            f"   └─ Fused context: {len(fused.fused_nodes)} unique nodes "
            f"| {len(fused.vector_nodes)} vector + "
            f"{len(fused.graph_nodes)} graph"
        )

        return fused

    # ── Step 2: Synthesize ────────────────────────────────────────────────────

    async def _step_synthesize(
        self,
        query: str,
        fused: FusedResult,
        log: list[str],
    ) -> str:
        """
        Generate initial answer from fused context.
        Runs LLM in executor to not block the event loop.
        """
        log.append("✍️  Step 2: Synthesizing answer from fused context...")

        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            _synthesize_answer,
            query,
            fused.context_text,
        )

        preview = answer[:120].replace("\n", " ")
        log.append(f"   └─ Answer draft: \"{preview}...\"")

        return answer

    # ── Step 3: Self-Correct ──────────────────────────────────────────────────

    async def _step_correct(
        self,
        query: str,
        answer: str,
        context: str,
        log: list[str],
        attempt: int = 1,
    ) -> CorrectionResult:
        """
        3-metric self-correction. Runs 3 sequential LLM scoring calls.
        Executed in executor to avoid blocking event loop.
        """
        log.append(
            f"🔎 Step {'3' if attempt == 1 else '5'}: "
            f"Self-Correction (attempt {attempt}/2)"
        )

        loop = asyncio.get_event_loop()
        correction = await loop.run_in_executor(
            None,
            run_self_correction,
            query,
            answer,
            context,
            attempt,
        )

        log.extend(correction.correction_log)
        return correction

    # ── Step 4: Re-Retrieve ───────────────────────────────────────────────────

    async def _step_reretrieve(
        self,
        requery: str,
        log: list[str],
    ) -> FusedResult:
        """
        Second retrieval pass using the targeted re-query string
        generated by the self-correction module.
        """
        log.append(f"🔄 Step 4: Re-Retrieval with targeted query")
        log.append(f"   └─ Re-query: \"{requery}\"")

        fused = await fuse_retrieve(requery, self.config, top_k=6)

        log.append(
            f"   └─ Re-retrieved: {len(fused.fused_nodes)} nodes"
        )
        return fused

    # ── Public Entry Point ────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        top_k: int = 6,
    ) -> AgentResponse:
        """
        Execute the full pipeline for one user query.

        Returns an AgentResponse regardless of success or failure —
        errors are captured in AgentResponse.error for graceful UI display.

        Args:
            query:  User's question string.
            top_k:  Results per database for RRF fusion.

        Returns:
            AgentResponse with answer, scores, and full thought log.
        """
        start_time = time.time()
        response = AgentResponse(query=query)
        log = response.thought_log

        log.append(f"🧠 Agent starting | query: \"{query[:80]}\"")
        log.append("─" * 50)

        try:
            # ── Attempt 1 ────────────────────────────────────────────────────

            # Step 1: RRF retrieval
            fused = await self._step_retrieve(query, log, top_k=top_k)
            response.fused_result = fused

            if not fused.fused_nodes:
                response.final_answer = (
                    "No relevant context found in the course materials. "
                    "Please ensure your lecture notes are ingested."
                )
                response.verdict = "INSUFFICIENT_CONTEXT"
                response.thought_log = log
                return response

            # Step 2: Synthesis
            answer = await self._step_synthesize(query, fused, log)

            # Step 3: Self-correction (attempt 1)
            correction = await self._step_correct(
                query, answer, fused.context_text, log, attempt=1
            )

            # ── Attempt 2 (if REQUERY triggered) ─────────────────────────────
            if correction.verdict == "REQUERY" and correction.requery_signal:
                log.append("─" * 50)
                response.attempt_count = 2

                # Step 4: Re-retrieve with targeted query
                fused2 = await self._step_reretrieve(
                    correction.requery_signal, log
                )

                # Merge original and re-retrieved context for richer synthesis
                merged_context = (
                    fused.context_text
                    + "\n\n--- ADDITIONAL CONTEXT (re-query) ---\n\n"
                    + fused2.context_text
                )

                # Step 5: Re-synthesize with merged context
                log.append("✍️  Step 5: Re-synthesizing with expanded context...")
                loop = asyncio.get_event_loop()
                answer2 = await loop.run_in_executor(
                    None,
                    _synthesize_answer,
                    query,
                    merged_context,
                )

                preview2 = answer2[:120].replace("\n", " ")
                log.append(f"   └─ Revised draft: \"{preview2}...\"")

                # Step 6: Final self-correction (attempt 2 — revises in-place)
                correction = await self._step_correct(
                    query, answer2, merged_context, log, attempt=2
                )

                # Update fused result to merged for attribution
                response.fused_result = fused2

            # ── Finalize ──────────────────────────────────────────────────────
            response.final_answer = correction.final_answer
            response.verdict = correction.verdict
            response.composite_score = correction.composite_score
            response.was_corrected = correction.was_corrected
            response.metric_scores = {
                name: {
                    "score": ms.score,
                    "reasoning": ms.reasoning,
                    "passed": ms.passed,
                }
                for name, ms in correction.metric_scores.items()
            }

            # Source attribution
            response.source_attributions = _extract_source_attributions(
                response.fused_result
            )

            response.route_summary = "hybrid-rrf"

        except Exception as e:
            error_msg = f"Workflow failed: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            response.error = error_msg
            response.final_answer = (
                "The agent encountered an unexpected error. "
                "Please check the terminal logs."
            )
            response.verdict = "ERROR"
            log.append(f"❌ {error_msg}")

        finally:
            response.latency_seconds = round(time.time() - start_time, 2)
            log.append("─" * 50)
            log.append(
                f"🏁 Pipeline complete | "
                f"verdict={response.verdict} | "
                f"score={response.composite_score:.2f}/10 | "
                f"latency={response.latency_seconds}s | "
                f"attempts={response.attempt_count}"
            )

        return response