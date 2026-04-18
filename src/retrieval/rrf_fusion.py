# src/retrieval/rrf_fusion.py
"""
Reciprocal Rank Fusion engine — the mathematical core of v2.0.

RRF Formula (Cormack et al., 2009):
    RRF(d) = Σ  1 / (k + rank_R(d))
           R ∈ {R_vec, R_graph}

Where:
    d       = a document/node
    R       = a ranked retrieval list
    rank_R  = position of d in list R (1-based)
    k       = smoothing constant (default 60)

Key properties:
    1. Rank-based — raw scores (cosine distance, graph hops) are discarded.
       This makes vector and graph results directly comparable.
    2. Additive consensus — a node appearing in BOTH lists gets contributions
       from each, naturally surfacing agreement between databases.
    3. Top-rank dampening — the k=60 constant prevents rank-1 from
       dominating (1/61 vs 1/62 is a smaller gap than 0.99 vs 0.50).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from src.retrieval.vector_retriever import RankedNode, retrieve_vector
from src.retrieval.graph_retriever import retrieve_graph
from src.utils import get_logger

logger = get_logger(__name__)

# Empirically optimal constant from original RRF paper
RRF_K: int = 60


@dataclass
class FusedResult:
    """
    Output of the RRF fusion engine.
    Handed directly to the LLM synthesizer and self-correction loop.
    """
    query: str
    fused_nodes: list[RankedNode]        # merged + re-ranked by RRF score
    vector_nodes: list[RankedNode]       # raw vector results (for eval)
    graph_nodes: list[RankedNode]        # raw graph results (for eval)
    context_text: str = ""               # concatenated text for LLM prompt
    retrieval_log: list[str] = field(default_factory=list)


def _compute_rrf_scores(
    vector_nodes: list[RankedNode],
    graph_nodes: list[RankedNode],
    k: int = RRF_K,
) -> list[RankedNode]:
    """
    Core RRF implementation.

    Algorithm:
        1. Build a score accumulator dict keyed by node_id
        2. For each list, add 1/(k + rank) to each node's score
        3. For nodes appearing in both lists, scores are summed
        4. Sort descending by final RRF score

    Nodes unique to one list still get scored (from that one list).
    Nodes in both lists get a higher combined score — this is the
    consensus signal that makes RRF powerful.

    Args:
        vector_nodes: Ranked results from ChromaDB (rank 1 = best)
        graph_nodes:  Ranked results from Neo4j (rank 1 = best)
        k:            Smoothing constant (default 60)

    Returns:
        Deduplicated, RRF-scored, descending-sorted list of RankedNode.
    """
    # Accumulator: node_id → (accumulated_rrf_score, node_object)
    scores: dict[str, list] = {}

    # We need a text-based dedup key since vector and graph nodes
    # have different id schemes but may contain overlapping content
    text_to_canonical_id: dict[str, str] = {}

    def get_canonical_id(node: RankedNode) -> str:
        """
        Map node to a canonical ID based on first 60 chars of text.
        This handles the case where the same source chunk appears
        in both vector and graph results with different node_ids.
        """
        text_key = node.text[:60].strip().lower()
        if text_key not in text_to_canonical_id:
            text_to_canonical_id[text_key] = node.node_id
        return text_to_canonical_id[text_key]

    # Process vector list
    for node in vector_nodes:
        cid = get_canonical_id(node)
        rrf_contribution = 1.0 / (k + node.rank)
        if cid not in scores:
            scores[cid] = [0.0, node]
        scores[cid][0] += rrf_contribution

    # Process graph list — same arithmetic, additive for overlapping nodes
    for node in graph_nodes:
        cid = get_canonical_id(node)
        rrf_contribution = 1.0 / (k + node.rank)
        if cid not in scores:
            scores[cid] = [0.0, node]
        scores[cid][0] += rrf_contribution

    # Build final list with RRF scores assigned
    fused: list[RankedNode] = []
    for cid, (rrf_score, node) in scores.items():
        node.rrf_score = round(rrf_score, 8)
        fused.append(node)

    # Sort descending by RRF score
    fused.sort(key=lambda n: n.rrf_score, reverse=True)

    # Re-assign final ranks
    for i, node in enumerate(fused, start=1):
        node.rank = i

    return fused


def _build_context_text(
    fused_nodes: list[RankedNode],
    max_nodes: int = 6,
) -> str:
    """
    Concatenate top-N fused node texts into a single LLM context string.
    Each chunk is labelled with its source and lecture for citation.
    """
    parts = []
    for node in fused_nodes[:max_nodes]:
        source_label = (
            f"[{node.source.upper()} | "
            f"{node.source_lecture} | "
            f"RRF={node.rrf_score:.4f}]"
        )
        parts.append(f"{source_label}\n{node.text}")

    return "\n\n---\n\n".join(parts)


def _build_retrieval_log(
    vector_nodes: list[RankedNode],
    graph_nodes: list[RankedNode],
    fused_nodes: list[RankedNode],
) -> list[str]:
    """Build the thought-process log entries shown in the Streamlit sidebar."""
    log = []

    log.append(
        f"🔀 Running parallel retrieval — "
        f"Vector (top-{len(vector_nodes)}) + Graph (top-{len(graph_nodes)})"
    )

    # Vector results summary
    log.append(f"📚 Vector results ({len(vector_nodes)}):")
    for n in vector_nodes[:3]:
        log.append(
            f"   [{n.rank}] sim={n.raw_score:.3f} | "
            f"{n.source_lecture} | \"{n.text[:80]}...\""
        )

    # Graph results summary
    log.append(f"🕸️  Graph results ({len(graph_nodes)}):")
    for n in graph_nodes[:3]:
        log.append(
            f"   [{n.rank}] | "
            f"{n.source_lecture} | \"{n.text[:80]}...\""
        )

    # RRF fusion outcome
    consensus = [n for n in fused_nodes if n.rrf_score > 1.0 / RRF_K * 1.5]
    log.append(
        f"⚡ RRF fusion → {len(fused_nodes)} unique nodes | "
        f"{len(consensus)} consensus node(s) appeared in both lists"
    )
    log.append("📊 Top-3 fused nodes after RRF:")
    for n in fused_nodes[:3]:
        log.append(
            f"   [rank={n.rank}] rrf={n.rrf_score:.4f} | "
            f"src={n.source} | \"{n.text[:80]}...\""
        )

    return log


async def fuse_retrieve(
    query: str,
    config: dict,
    top_k: int = 6,
) -> FusedResult:
    """
    Main entry point — parallel retrieval + RRF fusion.

    Executes vector and graph retrieval concurrently via asyncio.gather(),
    then applies RRF to produce a single merged ranked context.

    Args:
        query:  User question string.
        config: Loaded config.yaml dict.
        top_k:  Results per database before fusion.

    Returns:
        FusedResult with merged nodes and context string for the LLM.
    """
    logger.info(f"🔀 Fused retrieval starting | query='{query[:60]}...'")

    # ── Parallel retrieval ────────────────────────────────────────────────────
    vector_nodes, graph_nodes = await asyncio.gather(
        retrieve_vector(query, config, top_k=top_k),
        retrieve_graph(query, top_k=top_k),
        return_exceptions=False,
    )

    # Graceful degradation — if one source fails, continue with the other
    if not vector_nodes and not graph_nodes:
        logger.error("Both retrievers returned empty — cannot fuse.")
        return FusedResult(
            query=query,
            fused_nodes=[],
            vector_nodes=[],
            graph_nodes=[],
            context_text="No context could be retrieved from either database.",
            retrieval_log=["❌ Both vector and graph retrieval returned empty."],
        )

    if not vector_nodes:
        logger.warning("Vector retrieval empty — using graph results only.")
    if not graph_nodes:
        logger.warning("Graph retrieval empty — using vector results only.")

    # ── RRF Fusion ────────────────────────────────────────────────────────────
    fused_nodes = _compute_rrf_scores(
        vector_nodes=vector_nodes or [],
        graph_nodes=graph_nodes or [],
        k=RRF_K,
    )

    # ── Build outputs ─────────────────────────────────────────────────────────
    context_text = _build_context_text(fused_nodes, max_nodes=top_k)
    retrieval_log = _build_retrieval_log(vector_nodes, graph_nodes, fused_nodes)

    logger.info(
        f"✅ RRF fusion complete — {len(fused_nodes)} nodes | "
        f"top node: rrf={fused_nodes[0].rrf_score:.4f} "
        f"src={fused_nodes[0].source}"
        if fused_nodes else "✅ RRF fusion complete — 0 nodes"
    )

    return FusedResult(
        query=query,
        fused_nodes=fused_nodes,
        vector_nodes=vector_nodes or [],
        graph_nodes=graph_nodes or [],
        context_text=context_text,
        retrieval_log=retrieval_log,
    )