# src/ingestion.py

import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import nest_asyncio

from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.utils import load_config, get_logger, ensure_dirs

# Patch asyncio to allow nested event loops
nest_asyncio.apply()

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap() -> dict:
    load_dotenv()
    missing = [k for k in ["NEO4J_URI","NEO4J_USERNAME","NEO4J_PASSWORD"]
               if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}")
    config = load_config()
    ensure_dirs(config)
    logger.info("✅ Config and secrets loaded.")
    return config


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Configure LlamaIndex Settings
# ─────────────────────────────────────────────────────────────────────────────

def configure_llama_settings(config: dict) -> None:
    ollama_cfg = config["ollama"]
    Settings.llm = Ollama(
        model=ollama_cfg["llm_model"],
        base_url=ollama_cfg["base_url"],
        request_timeout=ollama_cfg["request_timeout"],
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=ollama_cfg["embed_model"],
        base_url=ollama_cfg["base_url"],
    )
    # Do NOT set chunk size here — we handle splitting manually below
    # so we can apply it consistently to both pipelines
    logger.info(
        f"✅ LLM: {ollama_cfg['llm_model']} | "
        f"Embeddings: {ollama_cfg['embed_model']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load + Pre-process Documents
# Key improvement: tag each chunk with its source filename as metadata.
# This lets the router later tell the user WHICH lecture the answer came from.
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(data_dir: str) -> list:
    data_path = Path(data_dir)
    found = (
        list(data_path.glob("*.txt")) +
        list(data_path.glob("*.pdf"))
    )
    if not found:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in '{data_dir}'."
        )

    logger.info(f"📂 Found {len(found)} file(s): {[f.name for f in found]}")

    documents = SimpleDirectoryReader(
        input_dir=data_dir,
        required_exts=[".txt", ".pdf"],
        recursive=False,
        # Attach filename as metadata to every chunk — shows up in answers
        filename_as_id=True,
    ).load_data()

    # Tag each document with a clean lecture name derived from filename
    for doc in documents:
        raw_name = Path(
            doc.metadata.get("file_name", "unknown")
        ).stem                              # e.g. "Lecture_03_WW1" → "Lecture 03 WW1"
        doc.metadata["lecture"] = raw_name.replace("_", " ").replace("-", " ")

    logger.info(f"✅ Loaded {len(documents)} document chunk(s) across {len(found)} file(s).")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Chunk documents with SentenceSplitter
# Splitting manually lets us reuse the SAME nodes for both KG and Vector.
# This avoids processing PDFs twice and halves ingestion time.
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list, config: dict) -> list:
    logger.info("✂️  Splitting documents into chunks...")
    splitter = SentenceSplitter(
        chunk_size=config["ingestion"]["chunk_size"],
        chunk_overlap=config["ingestion"]["chunk_overlap"],
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    logger.info(f"✅ Created {len(nodes)} chunks.")
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build Knowledge Graph → Push to Neo4j
# Using SimpleLLMPathExtractor — right tool for historical/theoretical text.
# It reliably extracts WHO/WHAT/WHEN/WHERE/WHY relationships from prose.
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_graph(nodes: list, config: dict) -> PropertyGraphIndex:
    logger.info("🔗 Connecting to Neo4j AuraDB...")
    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        database=config["neo4j"]["database"],
    )

    kg_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=config["ingestion"]["kg_max_triplets_per_chunk"],
        num_workers=1,
    )

    logger.info("🧠 Extracting KG triplets — this is the slow step...")
    pg_index = PropertyGraphIndex(
        nodes=nodes,                        # reuse pre-split nodes
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )

    logger.info("✅ Knowledge Graph written to Neo4j.")
    return pg_index


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Build Vector Index → Push to ChromaDB
# Reuses the same pre-split nodes — no double PDF parsing.
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_index(nodes: list, config: dict) -> VectorStoreIndex:
    chroma_path = config["paths"]["chroma_dir"]
    logger.info(f"📦 Connecting to ChromaDB at: {chroma_path}")

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection("course_materials")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("🔢 Generating embeddings...")
    vector_index = VectorStoreIndex(
        nodes=nodes,                        # reuse pre-split nodes
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("✅ Vector index written to ChromaDB.")
    return vector_index


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Incremental ingestion guard
# Checks which files are already in ChromaDB before processing.
# So if you add ONE new lecture PDF, only that file gets processed.
# ─────────────────────────────────────────────────────────────────────────────

def get_already_ingested(config: dict) -> set:
    """Return set of filenames already stored in ChromaDB."""
    try:
        chroma_client = chromadb.PersistentClient(
            path=config["paths"]["chroma_dir"]
        )
        collection = chroma_client.get_or_create_collection("course_materials")
        existing = collection.get(include=["metadatas"])
        ingested = set()
        for meta in existing.get("metadatas", []):
            if meta and meta.get("file_name"):
                ingested.add(meta["file_name"])
        return ingested
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(force_reingest: bool = False):
    logger.info("=" * 60)
    logger.info("   Agentic GraphRAG — Ingestion Pipeline")
    logger.info("=" * 60)

    config = bootstrap()
    configure_llama_settings(config)

    # Check what's already been ingested
    if not force_reingest:
        already_done = get_already_ingested(config)
        if already_done:
            logger.info(
                f"📋 Already ingested: {already_done}\n"
                "   Only new files will be processed. "
                "Use --force to re-ingest everything."
            )

    # Load documents
    all_documents = load_documents(config["paths"]["data_dir"])

    # Filter to only new files unless force flag is set
    if not force_reingest:
        already_done = get_already_ingested(config)
        new_documents = [
            d for d in all_documents
            if d.metadata.get("file_name", "") not in already_done
        ]
        if not new_documents:
            logger.info("✅ All files already ingested. Nothing to do.")
            logger.info("   Add new PDFs to data/raw/ and re-run to ingest them.")
            return
        logger.info(
            f"🆕 {len(new_documents)} new document(s) to ingest "
            f"(skipping {len(all_documents) - len(new_documents)} existing)."
        )
        documents_to_process = new_documents
    else:
        logger.info("⚠️  Force re-ingest enabled — processing all files.")
        documents_to_process = all_documents

    # Chunk once, reuse for both pipelines
    nodes = chunk_documents(documents_to_process, config)

    # Build both indexes from the same nodes
    build_knowledge_graph(nodes, config)
    build_vector_index(nodes, config)

    logger.info("=" * 60)
    logger.info("🎉 Ingestion complete!")
    logger.info(f"   Files processed: {len(documents_to_process)}")
    logger.info(f"   Chunks created : {len(nodes)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    run_ingestion(force_reingest=force)
