# src/ingestion.py

import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import nest_asyncio

# ── LlamaIndex core ──────────────────────────────────────────────────────────
from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

# ── LlamaIndex integrations ───────────────────────────────────────────────────
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Project utilities ─────────────────────────────────────────────────────────
from src.utils import load_config, get_logger, ensure_dirs

# Apply the nest_asyncio patch to allow nested event loops
nest_asyncio.apply()

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 ── Bootstrap: load secrets and config
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap() -> dict:
    """Load .env secrets and yaml config. Return merged config dict."""
    load_dotenv()

    required_env = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            "Check your .env file in the project root."
        )

    config = load_config()
    ensure_dirs(config)
    logger.info("✅ Config and secrets loaded successfully.")
    return config


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 ── Configure LlamaIndex global Settings
#           (replaces the old ServiceContext pattern in LlamaIndex 0.10.x)
# ─────────────────────────────────────────────────────────────────────────────

def configure_llama_settings(config: dict) -> None:
    """
    Set the global LLM and embedding model for all LlamaIndex operations.
    Using Ollama with local models — no API keys required.
    On M4 Mac, Ollama auto-detects Metal and uses GPU acceleration.
    """
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

    Settings.chunk_size = config["ingestion"]["chunk_size"]
    Settings.chunk_overlap = config["ingestion"]["chunk_overlap"]

    logger.info(
        f"✅ LlamaIndex configured — LLM: {ollama_cfg['llm_model']} | "
        f"Embeddings: {ollama_cfg['embed_model']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 ── Load documents from data/raw/
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(data_dir: str) -> list:
    """
    Read all .txt and .pdf files from the data directory.
    SimpleDirectoryReader handles both formats automatically.
    """
    data_path = Path(data_dir)
    if not any(data_path.glob("*.txt")) and not any(data_path.glob("*.pdf")):
        raise FileNotFoundError(
            f"No .txt or .pdf files found in '{data_dir}'.\n"
            "Add your course material files and re-run."
        )

    logger.info(f"📂 Loading documents from: {data_dir}")
    documents = SimpleDirectoryReader(
        input_dir=data_dir,
        required_exts=[".txt", ".pdf"],
        recursive=False,
    ).load_data()

    logger.info(f"✅ Loaded {len(documents)} document(s).")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 ── Build the Knowledge Graph → Push to Neo4j
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_graph(documents: list, config: dict) -> PropertyGraphIndex:
    """
    Uses LlamaIndex's PropertyGraphIndex with SimpleLLMPathExtractor.

    What happens internally:
      1. Documents are split into chunks (nodes).
      2. For each chunk, the LLM extracts (Subject → Relation → Object) triplets.
      3. Entities become Neo4j Nodes; relations become Neo4j Relationships.

    ⚠️  This step calls the LLM once per chunk — it will take several minutes
        for large document sets. This is expected. Watch the logs.
    """
    logger.info("🔗 Connecting to Neo4j AuraDB...")
    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        database=config["neo4j"]["database"],
    )

    # The extractor that prompts the LLM to find entities and relationships
    kg_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=config["ingestion"]["kg_max_triplets_per_chunk"],
        num_workers=1,          # Keep at 1 for local Ollama — avoids timeouts
    )

    logger.info(
        "🧠 Extracting Knowledge Graph triplets via LLM. "
        "This is the slow step — ~10-30s per chunk on M4..."
    )

    pg_index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )

    logger.info("✅ Knowledge Graph successfully written to Neo4j.")
    return pg_index


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 ── Build the Vector Index → Push to ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_index(documents: list, config: dict) -> VectorStoreIndex:
    """
    Embeds document chunks and stores them in a local ChromaDB collection.
    ChromaDB persists to disk at config.paths.chroma_dir — no server needed.
    """
    chroma_path = config["paths"]["chroma_dir"]
    logger.info(f"📦 Initialising ChromaDB at: {chroma_path}")

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection("course_materials")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("🔢 Generating embeddings and writing to ChromaDB...")
    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("✅ Vector index successfully written to ChromaDB.")
    return vector_index


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion():
    logger.info("=" * 60)
    logger.info("   Agentic GraphRAG — Ingestion Pipeline Starting")
    logger.info("=" * 60)

    config = bootstrap()
    configure_llama_settings(config)
    documents = load_documents(config["paths"]["data_dir"])

    # Run both pipelines on the same loaded documents
    build_knowledge_graph(documents, config)
    build_vector_index(documents, config)

    logger.info("=" * 60)
    logger.info("🎉 Ingestion complete! Both databases are populated.")
    logger.info("   → Check Neo4j AuraDB browser to visualise your graph.")
    logger.info("   → ChromaDB persisted locally at: " + config["paths"]["chroma_dir"])
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
