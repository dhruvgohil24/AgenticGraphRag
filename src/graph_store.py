# src/graph_store.py

import os
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from src.utils import load_config, get_logger

logger = get_logger(__name__)


def load_graph_index(config: dict) -> PropertyGraphIndex:
    """
    Reconnect to the existing Neo4j graph WITHOUT re-running extraction.
    This is what the router calls at startup to query the graph.
    """
    load_dotenv()
    logger.info("🔗 Reconnecting to Neo4j property graph store...")

    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        database=config["neo4j"]["database"],
    )

    # from_existing() reattaches to the populated graph — no LLM calls here
    pg_index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
    )

    logger.info("✅ Graph index loaded from Neo4j.")
    return pg_index


def load_vector_index(config: dict) -> VectorStoreIndex:
    """
    Reconnect to the persisted ChromaDB collection WITHOUT re-embedding.
    """
    chroma_path = config["paths"]["chroma_dir"]
    logger.info(f"📦 Reconnecting to ChromaDB at: {chroma_path}")

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = chroma_client.get_or_create_collection("course_materials")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    logger.info("✅ Vector index loaded from ChromaDB.")
    return vector_index