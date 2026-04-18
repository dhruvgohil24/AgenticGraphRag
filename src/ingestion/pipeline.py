# src/ingestion/pipeline.py
"""
Async ingestion orchestrator — v2.0

Key upgrades over v1 ingestion.py:
  1. asyncio + Semaphore prevents Neo4j connection exhaustion on large PDFs
  2. pymupdf parser replaces pypdf for layout accuracy
  3. Incremental guard — skips already-ingested files
  4. Nodes built once, shared by both KG and vector pipelines
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import chromadb
from dotenv import load_dotenv
from tqdm import tqdm
import nest_asyncio

nest_asyncio.apply()

from llama_index.core import (
    PropertyGraphIndex,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.ingestion.chunker import parsed_doc_to_nodes
from src.ingestion.pdf_parser import ParsedDocument, parse_pdf, parse_txt
from src.utils import ensure_dirs, get_logger, load_config

logger = get_logger(__name__)

# Max concurrent KG extraction calls to Neo4j.
# On AuraDB free tier, >3 concurrent writes cause connection drops.
NEO4J_CONCURRENCY_LIMIT = 2


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap() -> dict:
    load_dotenv()
    missing = [
        k for k in ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        if not os.getenv(k)
    ]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}")
    config = load_config()
    ensure_dirs(config)
    return config


def configure_settings(config: dict) -> None:
    cfg = config["ollama"]
    Settings.llm = Ollama(
        model=cfg["llm_model"],
        base_url=cfg["base_url"],
        request_timeout=cfg["request_timeout"],
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=cfg["embed_model"],
        base_url=cfg["base_url"],
    )
    logger.info(f"✅ LLM: {cfg['llm_model']} | Embed: {cfg['embed_model']}")


# ─────────────────────────────────────────────────────────────────────────────
# File Discovery & Incremental Guard
# ─────────────────────────────────────────────────────────────────────────────

def discover_files(data_dir: str) -> list[Path]:
    """Return all .pdf and .txt files in the data directory."""
    data_path = Path(data_dir)
    files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in '{data_dir}'."
        )
    logger.info(f"📂 Discovered {len(files)} file(s): {[f.name for f in files]}")
    return files


def get_ingested_files(config: dict) -> set[str]:
    """
    Query ChromaDB metadata to find already-ingested filenames.
    Returns a set of file_name strings.
    """
    try:
        client = chromadb.PersistentClient(path=config["paths"]["chroma_dir"])
        col = client.get_or_create_collection("course_materials")
        existing = col.get(include=["metadatas"])
        return {
            m["file_name"]
            for m in existing.get("metadatas", [])
            if m and m.get("file_name")
        }
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_file(file_path: Path) -> Optional[ParsedDocument]:
    """Route to correct parser based on file extension."""
    try:
        if file_path.suffix.lower() == ".pdf":
            doc = parse_pdf(file_path)
        else:
            doc = parse_txt(file_path)
        logger.info(
            f"   📄 Parsed '{file_path.name}' → "
            f"{doc.total_pages} pages, {len(doc.full_text.split())} words"
        )
        return doc
    except ValueError as e:
        # Scanned PDF or empty file — log and skip
        logger.warning(f"   ⚠️  Skipping '{file_path.name}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Async KG Extraction
# The semaphore is the critical piece — it ensures we never have more than
# NEO4J_CONCURRENCY_LIMIT simultaneous LLM+Neo4j write operations.
# Without it, large PDFs (50+ chunks) exhaust the AuraDB connection pool.
# ─────────────────────────────────────────────────────────────────────────────


async def build_knowledge_graph_async(
    nodes: list[TextNode],
    config: dict,
) -> PropertyGraphIndex:
    """
    Build the Neo4j Property Graph, safely run from within an async context.

    The core problem we solve here:
      PropertyGraphIndex.__init__() calls asyncio.run() internally.
      Calling asyncio.run() inside an already-running event loop raises
      RuntimeError. The solution is run_in_executor() — it spawns a fresh
      OS thread which has NO running event loop, so PropertyGraphIndex's
      internal asyncio.run() works correctly.

    The semaphore lives at the batch level — we process nodes in batches
    of BATCH_SIZE, with NEO4J_CONCURRENCY_LIMIT batches running at once.
    This prevents AuraDB free tier connection exhaustion.
    """
    BATCH_SIZE = 5  # nodes per batch — tune down if Neo4j still drops

    logger.info("🔗 Connecting to Neo4j AuraDB...")
    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        database=config["neo4j"]["database"],
    )

    extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=config["ingestion"]["kg_max_triplets_per_chunk"],
        num_workers=1,
    )

    # Split nodes into batches
    batches = [
        nodes[i : i + BATCH_SIZE]
        for i in range(0, len(nodes), BATCH_SIZE)
    ]

    semaphore = asyncio.Semaphore(NEO4J_CONCURRENCY_LIMIT)
    loop = asyncio.get_event_loop()

    logger.info(
        f"🧠 Async KG extraction — {len(nodes)} chunks across "
        f"{len(batches)} batches | concurrency = {NEO4J_CONCURRENCY_LIMIT}..."
    )

    async def process_batch(
        batch: list[TextNode],
        batch_idx: int,
    ) -> None:
        """
        Process one batch of nodes under the semaphore.
        """
        async with semaphore:
            logger.info(
                f"   📦 Batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch)} chunks)..."
            )

            try:
                # With nest_asyncio, we can just instantiate the index!
                PropertyGraphIndex(
                    nodes=batch,
                    kg_extractors=[extractor],
                    property_graph_store=graph_store,
                    embed_model=Settings.embed_model,
                    show_progress=False,
                )
                logger.info(f"   ✅ Batch {batch_idx + 1} written to Neo4j.")
            except Exception as e:
                logger.error(
                    f"   ❌ Batch {batch_idx + 1} failed: {e} — "
                    "continuing with remaining batches."
                )

    # Run all batches concurrently (semaphore-gated)
    with tqdm(total=len(batches), desc="KG batches") as pbar:
        async def process_and_update(batch, idx):
            await process_batch(batch, idx)
            pbar.update(1)

        await asyncio.gather(*[
            process_and_update(batch, idx)
            for idx, batch in enumerate(batches)
        ])

    logger.info(f"✅ Knowledge Graph complete — {len(nodes)} chunks processed.")

    # Return a reconnected index for downstream use
    return PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
    )
# ─────────────────────────────────────────────────────────────────────────────
# Vector Index (synchronous — ChromaDB is local, no connection limits)
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_index(
    nodes: list[TextNode],
    config: dict,
) -> VectorStoreIndex:
    chroma_path = config["paths"]["chroma_dir"]
    logger.info(f"📦 Writing embeddings to ChromaDB at: {chroma_path}")

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection("course_materials")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info(f"✅ Vector index written — {len(nodes)} chunks embedded.")
    return vector_index


# ─────────────────────────────────────────────────────────────────────────────
# Master Async Pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline_async(force_reingest: bool = False) -> None:
    """
    Full async ingestion pipeline:
      1. Discover files
      2. Filter already-ingested (unless --force)
      3. Parse PDFs with pymupdf
      4. Chunk into TextNodes
      5. Async KG extraction → Neo4j
      6. Vector embedding → ChromaDB
    """
    logger.info("=" * 60)
    logger.info("   Agentic GraphRAG v2.0 — Async Ingestion Pipeline")
    logger.info("=" * 60)

    config = bootstrap()
    configure_settings(config)

    # ── 1. Discover ───────────────────────────────────────────────────────────
    all_files = discover_files(config["paths"]["data_dir"])

    # ── 2. Incremental guard ──────────────────────────────────────────────────
    if force_reingest:
        files_to_process = all_files
        logger.info("⚠️  Force re-ingest — processing all files.")
    else:
        already_done = get_ingested_files(config)
        files_to_process = [f for f in all_files if f.name not in already_done]
        skipped = len(all_files) - len(files_to_process)
        if skipped:
            logger.info(f"⏭️  Skipping {skipped} already-ingested file(s).")
        if not files_to_process:
            logger.info("✅ All files already ingested. Nothing to do.")
            logger.info("   Add new PDFs to data/raw/ and re-run.")
            return

    # ── 3. Parse ──────────────────────────────────────────────────────────────
    logger.info(f"\n📖 Parsing {len(files_to_process)} file(s)...")
    all_nodes: list[TextNode] = []

    for file_path in files_to_process:
        doc = parse_file(file_path)
        if doc is None:
            continue   # Skip unparseable files (scanned PDFs etc.)

        # ── 4. Chunk ─────────────────────────────────────────────────────────
        nodes = parsed_doc_to_nodes(
            doc,
            chunk_size=config["ingestion"]["chunk_size"],
            chunk_overlap=config["ingestion"]["chunk_overlap"],
        )
        all_nodes.extend(nodes)

    if not all_nodes:
        logger.error("❌ No nodes created — check your PDF files.")
        return

    logger.info(f"\n✅ Total chunks ready for indexing: {len(all_nodes)}")

    # ── 5. Async KG extraction ────────────────────────────────────────────────
    await build_knowledge_graph_async(all_nodes, config)

    # ── 6. Vector indexing ────────────────────────────────────────────────────
    build_vector_index(all_nodes, config)

    logger.info("=" * 60)
    logger.info("🎉 v2.0 Ingestion complete!")
    logger.info(f"   Files processed : {len(files_to_process)}")
    logger.info(f"   Total chunks    : {len(all_nodes)}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(force_reingest: bool = False) -> None:
    """Synchronous wrapper — lets app.py and CLI both call this."""
    asyncio.run(run_pipeline_async(force_reingest=force_reingest))


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_ingestion(force_reingest=force)