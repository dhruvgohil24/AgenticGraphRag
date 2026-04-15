# src/router.py

"""
The Agentic Router — Heart of the GraphRAG system.

Decision Logic:
┌─────────────────────────────────────────────────────────────┐
│  User Query                                                  │
│       │                                                      │
│       ▼                                                      │
│  [LLM Classifier]  ──► "vector" or "graph"                  │
│       │                                                      │
│  ┌────┴────┐                                                 │
│  │         │                                                 │
│  ▼         ▼                                                 │
│ Vector    Graph                                              │
│  DB        DB                                                │
│ (broad)  (multi-hop)                                         │
│  └────┬────┘                                                 │
│       ▼                                                      │
│  Retrieved Context                                           │
│       │                                                      │
│       ▼                                                      │
│  [LLM Synthesizer] ──► Final Answer + Thought Log           │
└─────────────────────────────────────────────────────────────┘
"""
# Add this with the other imports at the top
import chromadb
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from llama_index.core import Settings
from llama_index.core.indices.property_graph import (
    PropertyGraphIndex,
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine

from src.utils import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data contract: everything the Streamlit UI needs lives in this object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouterResponse:
    """
    Structured output from the router.
    The UI reads `answer` for the chat bubble and `thought_log` for the sidebar.
    """
    query: str
    route_taken: Literal["vector", "graph", "hybrid"] = "vector"
    answer: str = ""
    retrieved_context: str = ""
    thought_log: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Query Classifier
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """\
You are an expert query router for an educational Q&A system.

Your job is to classify a student's question into exactly one of two categories:

CATEGORY "vector":
- Broad conceptual questions needing summarization
- "What is X?", "Explain Y", "Give an overview of Z"
- Questions where a general passage of text is the best answer

CATEGORY "graph":
- Questions about specific relationships between entities
- "How does X relate to Y?", "What causes Z?", "What are the steps from A to B?"
- Questions requiring multi-hop reasoning across multiple facts
- "Who/What connects X and Y?"

Question: {query}

Reply with ONLY one word — either "vector" or "graph". No explanation.
"""

def classify_query(query: str) -> tuple[Literal["vector", "graph"], str]:
    """
    Ask the LLM to classify the query.
    Returns (route, reasoning_note_for_log).
    Falls back to "vector" if classification is ambiguous.
    """
    prompt = CLASSIFIER_PROMPT.format(query=query)

    try:
        response = Settings.llm.complete(prompt)
        raw = response.text.strip().lower()

        # Be robust — extract "vector" or "graph" even if LLM adds a word
        if "graph" in raw:
            return "graph", f"LLM classified as GRAPH-RELATIONAL (raw: '{raw}')"
        elif "vector" in raw:
            return "vector", f"LLM classified as VECTOR-SEMANTIC (raw: '{raw}')"
        else:
            # Fallback: if LLM produced gibberish, default to vector
            logger.warning(f"Ambiguous classification response: '{raw}' — defaulting to vector.")
            return "vector", f"Classification ambiguous (raw: '{raw}') → defaulted to VECTOR"

    except Exception as e:
        logger.error(f"Classifier LLM call failed: {e}")
        return "vector", f"Classifier failed ({e}) → defaulted to VECTOR"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2A — Vector Retrieval Path (Direct ChromaDB Query — bypasses LlamaIndex
#            wrapper bug where empty where={} filter crashes ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_from_vector(
    query: str,
    vector_index: VectorStoreIndex,
    config: dict,
    top_k: int = 4,
) -> tuple[str, str]:
    """
    Queries ChromaDB directly using the embedding model.
    Steps:
      1. Embed the query using Ollama nomic-embed-text
      2. Query ChromaDB collection for top-k similar chunks
      3. Feed raw chunks as context to LLM for synthesis
    """
    # Step 1 — Embed the query
    embed_result = Settings.embed_model.get_text_embedding(query)

    # Step 2 — Query ChromaDB directly (no LlamaIndex wrapper, no where={} bug)
    chroma_client = chromadb.PersistentClient(path=config["paths"]["chroma_dir"])
    collection = chroma_client.get_or_create_collection("course_materials")

    results = collection.query(
        query_embeddings=[embed_result],
        n_results=top_k,
        include=["documents", "distances"],
    )

    # Step 3 — Format retrieved chunks
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return "No relevant context found in the vector store.", "No documents retrieved."

    # Build context string for the LLM
    context_parts = []
    context_display_parts = []

    for i, (doc, dist) in enumerate(zip(documents, distances)):
        context_parts.append(f"[Chunk {i+1}]:\n{doc}")
        score = round(1 - dist, 3)   # convert distance → similarity score
        snippet = doc[:200].replace("\n", " ")
        context_display_parts.append(f"  [similarity={score}] \"{snippet}...\"")

    context_for_llm = "\n\n".join(context_parts)
    context_display = "\n".join(context_display_parts)

    # Step 4 — Synthesize answer using LLM
    synthesis_prompt = f"""\
You are a helpful educational assistant. Using ONLY the context below, \
answer the student's question accurately and concisely.
If the context does not contain enough information, say so clearly.

CONTEXT:
{context_for_llm}

QUESTION: {query}

ANSWER:"""

    response = Settings.llm.complete(synthesis_prompt)
    return str(response).strip(), context_display
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2B — Graph Retrieval Path
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_from_graph(
    query: str,
    graph_index: PropertyGraphIndex,
) -> tuple[str, str]:
    """
    Multi-hop traversal over Neo4j via LlamaIndex's graph retriever.

    Uses LLMSynonymRetriever — it expands the query into entity synonyms,
    looks them up in the graph, and traverses outbound relationships to
    collect multi-hop context. Much more powerful than a simple lookup.

    Returns (synthesized_answer, raw_triplets_found).
    """
    # Build a retriever that does entity-synonym expansion + graph traversal
    retriever = graph_index.as_retriever(
        sub_retrievers=[
            LLMSynonymRetriever(
                graph_index.property_graph_store,
                llm=Settings.llm,
                include_text=True,
            ),
            VectorContextRetriever(
                graph_index.property_graph_store,
                embed_model=Settings.embed_model,
                include_text=True,
            ),
        ]
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=Settings.llm,
    )
    response = query_engine.query(query)

    # Collect graph nodes/triplets for the thought log
    context_snippets = []
    for node in response.source_nodes:
        snippet = node.get_text()[:200].replace("\n", " ")
        context_snippets.append(f"  [graph-node] \"{snippet}...\"")

    context_display = (
        "\n".join(context_snippets) if context_snippets
        else "No graph paths found — answer may be based on text context."
    )
    return str(response), context_display


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Master Router Function (called by Streamlit app)
# ─────────────────────────────────────────────────────────────────────────────

def route_query(
    query: str,
    graph_index: PropertyGraphIndex,
    vector_index: VectorStoreIndex,
    config: dict,
) -> RouterResponse:
    """
    Main entry point for the router. Called once per user message.

    Returns a RouterResponse containing:
      - The final answer
      - The route taken (vector / graph)
      - A structured thought_log list for the Streamlit sidebar
    """
    result = RouterResponse(query=query)
    start_time = time.time()

    # ── 1. Classify ──────────────────────────────────────────────────────────
    result.thought_log.append("🔍 Step 1: Analysing query intent...")
    route, classification_note = classify_query(query)
    result.route_taken = route
    result.thought_log.append(f"   └─ {classification_note}")

    # ── 2. Retrieve ──────────────────────────────────────────────────────────
    try:
        if route == "vector":
            result.thought_log.append("📚 Step 2: Routing to Vector DB (ChromaDB)...")
            result.thought_log.append("   └─ Running semantic similarity search (top-4 chunks)...")
            answer, context = retrieve_from_vector(query, vector_index, config)

        else:  # graph
            result.thought_log.append("🕸️  Step 2: Routing to Graph DB (Neo4j)...")
            result.thought_log.append("   └─ Expanding entities via LLM synonym retriever...")
            result.thought_log.append("   └─ Traversing knowledge graph for multi-hop paths...")
            answer, context = retrieve_from_graph(query, graph_index)

        result.answer = answer
        result.retrieved_context = context
        result.thought_log.append(f"📄 Step 3: Context retrieved:\n{context}")

    except Exception as e:
        error_msg = f"Retrieval failed on route='{route}': {e}"
        logger.error(error_msg)
        result.error = error_msg
        result.answer = (
            "I encountered an error while retrieving context. "
            "Please check the logs for details."
        )
        result.thought_log.append(f"❌ ERROR: {error_msg}")

    # ── 3. Finalise ──────────────────────────────────────────────────────────
    result.latency_seconds = round(time.time() - start_time, 2)
    result.thought_log.append(
        f"✅ Done — Route: {result.route_taken.upper()} | "
        f"Latency: {result.latency_seconds}s"
    )

    logger.info(
        f"Query routed → {result.route_taken.upper()} | "
        f"{result.latency_seconds}s | query='{query[:60]}...'"
    )
    return result