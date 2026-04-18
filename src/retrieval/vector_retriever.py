# src/retrieval/vector_retriever.py
"""
Async vector retriever against ChromaDB.
Returns a ranked list of RankedNode objects for RRF fusion.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import Settings

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RankedNode:
    """
    Unified retrieval result — works for both vector and graph results.
    RRF operates on rank positions, so raw scores are stored but not used
    for fusion arithmetic directly.
    """
    node_id: str
    text: str
    source: str                          # "vector" | "graph"
    rank: int = 0                        # 1-based rank within its source list
    raw_score: float = 0.0               # cosine similarity or graph hop score
    rrf_score: float = 0.0               # computed by RRF fusion
    metadata: dict = field(default_factory=dict)

    @property
    def source_lecture(self) -> str:
        return self.metadata.get("source_lecture", "Unknown Lecture")

    @property
    def approx_page(self) -> int:
        return self.metadata.get("approx_page", 0)


async def retrieve_vector(
    query: str,
    config: dict,
    top_k: int = 6,
) -> list[RankedNode]:
    """
    Async vector retrieval from ChromaDB.

    Embeds the query, queries ChromaDB for top_k similar chunks,
    returns them as ranked RankedNode objects.

    Args:
        query:  User question string.
        config: Loaded config.yaml dict.
        top_k:  Number of results to retrieve.

    Returns:
        List of RankedNode sorted by descending similarity (rank 1 = best).
    """
    loop = asyncio.get_event_loop()

    def _query_chroma() -> list[RankedNode]:
        # Embed query
        embedding = Settings.embed_model.get_text_embedding(query)

        # Query ChromaDB
        client = chromadb.PersistentClient(
            path=config["paths"]["chroma_dir"]
        )
        collection = client.get_or_create_collection("course_materials")

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "distances", "metadatas"],
        )

        docs      = results.get("documents", [[]])[0]
        distances = results.get("distances",  [[]])[0]
        metadatas = results.get("metadatas",  [[]])[0]

        ranked: list[RankedNode] = []
        for rank_idx, (doc, dist, meta) in enumerate(
            zip(docs, distances, metadatas), start=1
        ):
            # ChromaDB returns L2 distance → convert to similarity
            similarity = round(1.0 / (1.0 + dist), 6)
            ranked.append(RankedNode(
                node_id=f"vec_{rank_idx}_{hash(doc[:40])}",
                text=doc,
                source="vector",
                rank=rank_idx,
                raw_score=similarity,
                metadata=meta or {},
            ))

        return ranked

    try:
        nodes = await loop.run_in_executor(None, _query_chroma)
        logger.info(f"📚 Vector retrieval → {len(nodes)} chunks retrieved.")
        return nodes
    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")
        return []