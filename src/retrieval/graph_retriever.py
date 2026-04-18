# src/retrieval/graph_retriever.py
"""
Async graph retriever against Neo4j via LlamaIndex PropertyGraphIndex.
Returns a ranked list of RankedNode objects for RRF fusion.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex

from src.retrieval.vector_retriever import RankedNode
from src.utils import get_logger

logger = get_logger(__name__)


def _clean_graph_text(raw: str) -> str:
    """
    Strip the LlamaIndex KG prefix and Python dict metadata
    that Neo4j nodes carry after extraction.
    Before: "Here are some facts extracted from the provided text:
             Term X ({'file_path': '...', 'triplet_source_id': '...'})"
    After:  "Term X"
    """
    # Remove the boilerplate prefix
    if "Here are some facts extracted" in raw:
        raw = raw.split("Here are some facts extracted from the provided text:")[-1]

    # Remove Python dict metadata blobs: ({'key': 'value', ...})
    raw = re.sub(r"\(\{.*?\}\)", "", raw, flags=re.DOTALL)

    # Remove leftover brackets and normalize whitespace
    raw = re.sub(r"[\(\)\{\}]", "", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _build_graph_store() -> Neo4jPropertyGraphStore:
    """Create a Neo4j connection. Called inside executor threads."""
    load_dotenv()
    
    from src.utils import load_config
    config = load_config()

    return Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        database=config["neo4j"]["database"],
    )


async def retrieve_graph(
    query: str,
    top_k: int = 6,
) -> list[RankedNode]:
    """
    Async graph retrieval from Neo4j.

    Uses LLMSynonymRetriever (entity expansion) and
    VectorContextRetriever (embedding-based graph node search) in parallel,
    then deduplicates and ranks results.

    Args:
        query:  User question string.
        top_k:  Max number of graph nodes to return.

    Returns:
        List of RankedNode sorted by traversal relevance (rank 1 = best).
    """
    loop = asyncio.get_event_loop()

    def _query_graph() -> list[RankedNode]:
        graph_store = _build_graph_store()

        pg_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
        )

        # Two complementary retrievers:
        # LLMSynonymRetriever — expands query to entity synonyms, does keyword match
        # VectorContextRetriever — embedding match against graph node text
        retriever = pg_index.as_retriever(
            sub_retrievers=[
                LLMSynonymRetriever(
                    pg_index.property_graph_store,
                    llm=Settings.llm,
                    include_text=True,
                ),
                VectorContextRetriever(
                    pg_index.property_graph_store,
                    embed_model=Settings.embed_model,
                    include_text=True,
                ),
            ]
        )

        raw_nodes = retriever.retrieve(query)

        # Deduplicate by text content (both sub-retrievers may return same node)
        seen_texts: set[str] = set()
        deduped = []
        for node in raw_nodes:
            text_key = node.get_text()[:80]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduped.append(node)

        # Limit to top_k after dedup
        deduped = deduped[:top_k]

        ranked: list[RankedNode] = []
        for rank_idx, node in enumerate(deduped, start=1):
            raw_text = node.get_text()
            clean_text = _clean_graph_text(raw_text)

            # Extract metadata if available
            meta = {}
            if hasattr(node, "metadata"):
                meta = node.metadata or {}

            score = node.score if hasattr(node, "score") and node.score else 0.0

            ranked.append(RankedNode(
                node_id=f"graph_{rank_idx}_{hash(clean_text[:40])}",
                text=clean_text,
                source="graph",
                rank=rank_idx,
                raw_score=float(score),
                metadata=meta,
            ))

        return ranked

    try:
        nodes = await loop.run_in_executor(None, _query_graph)
        logger.info(f"🕸️  Graph retrieval → {len(nodes)} nodes retrieved.")
        return nodes
    except Exception as e:
        logger.error(f"Graph retrieval failed: {e}")
        return []