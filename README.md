<div align="center">

# Agentic GraphRAG
### Neural-Symbolic Retrieval Engine for Educational Question-Answering

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10.68-purple)](https://llamaindex.ai)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-green?logo=neo4j)](https://neo4j.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-orange)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Llama--3--8B-black?logo=ollama)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*Guided by Prof. Shyamanta M. Hazarika — Indian Institute of Technology Guwahati*

[Overview](#overview) · [Architecture](#architecture) · [Features](#features) · [Setup](#setup) · [Usage](#usage) · [Evaluation](#evaluation) · [Project Structure](#project-structure)

</div>

---

## Overview

Agentic GraphRAG is an end-to-end, locally-hosted retrieval-augmented generation system 
that answers questions over academic course materials with **provably grounded responses**. 
Unlike standard RAG pipelines that query a single vector database, this system maintains 
**two complementary knowledge representations** of the same source material — a vector 
index for semantic similarity and a property knowledge graph for relational reasoning — 
and queries both simultaneously on every user request.

The system was built to solve a fundamental problem with LLM-based educational assistants: 
hallucination. By grounding every answer in retrieved context and running a three-stage 
verification loop before returning a response, the system can flag and correct answers 
that introduce facts not present in the course material.

**Everything runs locally.** No OpenAI API keys. No paid inference endpoints. 
The entire pipeline — ingestion, retrieval, synthesis, and evaluation — runs on a 
consumer MacBook using Ollama with Llama-3-8B.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              HYBRID RRF RETRIEVAL (parallel)                    │
│                                                                 │
│   ┌──────────────────┐         ┌──────────────────────┐        │
│   │   ChromaDB       │         │   Neo4j AuraDB        │        │
│   │   Vector Index   │         │   Property Graph      │        │
│   │                  │         │                       │        │
│   │ nomic-embed-text │         │ LLMSynonymRetriever   │        │
│   │ cosine similarity│         │ VectorContextRetriever│        │
│   └────────┬─────────┘         └──────────┬────────────┘        │
│            │  ranked list R₁              │  ranked list R₂     │
│            └──────────────┬───────────────┘                     │
│                           │                                     │
│                    RRF Fusion                                   │
│             score(d) = Σ 1/(60 + rank_R(d))                    │
│                           │                                     │
│                    Merged Context                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM SYNTHESIS (Llama-3-8B)                    │
│                   Initial answer draft                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              DSPy-INSPIRED SELF-CORRECTION LOOP                 │
│                                                                 │
│   Metric 1: Groundedness      (weight 0.50)                     │
│   Metric 2: Context Relevance (weight 0.30)                     │
│   Metric 3: Answer Completeness (weight 0.20)                   │
│                                                                 │
│   composite ≥ 6.5 → APPROVED                                   │
│   composite < 6.5 → Re-query → Re-synthesize → Final correct   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│            STREAMLIT UI — AgentResponse                         │
│   Answer · Metric Scorecard · Source Attribution · Thought Log  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

**Hybrid RRF Retrieval**
Executes vector and graph retrieval in parallel using `asyncio.gather()`. 
Results from both databases are merged using the Reciprocal Rank Fusion algorithm 
(Cormack et al., 2009), which operates purely on rank position — making cosine 
similarities and graph traversal scores directly comparable without normalization.

**Asynchronous Ingestion Pipeline**
Large PDF corpora are ingested using an `asyncio` + `Semaphore`-gated batch 
processor that prevents Neo4j AuraDB free-tier connection exhaustion. PyMuPDF 
extracts text with layout preservation, handling multi-column slides and 
suppressing blank pages and artifacts automatically.

**DSPy-Inspired Three-Metric Self-Correction**
Inspired by the DSPy "Signature + Metric" pattern, three independent focused 
LLM calls evaluate Groundedness, Context Relevance, and Answer Completeness 
before any answer reaches the user. Composite scores below the threshold (6.5/10) 
trigger a targeted re-query — the system diagnoses *which* metric failed and 
generates a more specific search query before re-synthesizing.

**Incremental Ingestion Guard**
Re-running the ingestion script only processes files not yet present in ChromaDB. 
Add one new lecture PDF to `data/raw/` and re-run — only that file is processed.

**Quantitative Evaluation Suite**
A standalone evaluation runner measures four Ragas-compatible metrics — 
Faithfulness, Answer Relevancy, Context Precision, and Context Recall — 
over a configurable question bank with ground-truth answers. Results are written 
to a timestamped CSV for reproducible benchmarking.

**Fully Local — No API Keys Required**
Every component (ingestion LLM, embedding model, synthesis, self-correction, 
evaluation) uses Llama-3-8B and nomic-embed-text served locally via Ollama. 
Runs with Metal GPU acceleration on Apple Silicon.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.11 | Type-hinted, async-native |
| LLM | Llama-3-8B via Ollama | Synthesis, extraction, scoring |
| Embeddings | nomic-embed-text via Ollama | Semantic similarity |
| Orchestration | LlamaIndex 0.10.x | PropertyGraph, retrieval pipelines |
| Graph Database | Neo4j AuraDB (free tier) | Entity-relationship storage |
| Vector Database | ChromaDB 0.4.24 | Embedding persistence |
| PDF Parsing | PyMuPDF (fitz) | Layout-preserving text extraction |
| Fusion Algorithm | Reciprocal Rank Fusion | Cross-DB result merging |
| Self-Correction | DSPy-inspired metric loop | Hallucination elimination |
| Evaluation | Custom Ragas-equivalent | Quantitative benchmarking |
| Frontend | Streamlit 1.37 | Chat UI with metric scorecard |

---

## Setup

### Prerequisites

Before cloning this repository, ensure the following are installed and running 
on your machine.

**1. Python 3.11**

```bash
# macOS (Homebrew)
brew install python@3.11
python3.11 --version   # should print Python 3.11.x
```

> ⚠️ Python 3.12+ and 3.14 are not compatible with the Pydantic v1 shim used 
> internally by LlamaIndex 0.10.x. Use 3.11 exactly.

**2. Ollama with required models**

Download Ollama from [ollama.com](https://ollama.com), then pull both models:

```bash
ollama pull llama3            # ~4.7 GB — main reasoning LLM
ollama pull nomic-embed-text  # ~274 MB — embedding model
ollama serve                  # keep this running in a separate terminal
```

Verify both are available:

```bash
ollama list
# NAME                    ID              SIZE    MODIFIED
# llama3:latest           ...             4.7 GB  ...
# nomic-embed-text:latest ...             274 MB  ...
```

**3. Neo4j AuraDB Free Instance**

1. Go to [console.neo4j.io](https://console.neo4j.io) and sign up for a free account
2. Click **New Instance → AuraDB Free**
3. When the credentials screen appears, copy your `URI`, `Username`, and `Password`
   immediately — the password is shown only once
4. Keep this tab open — you will need to paste these into `.env` in the next step

---

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/agentic-graphrag.git
cd agentic-graphrag

# Create and activate a Python 3.11 virtual environment
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Verify Python version inside the venv
python --version                  # must print Python 3.11.x

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import llama_index.core; print('LlamaIndex OK')"
python -c "import fitz; print('PyMuPDF OK')"
python -c "import chromadb; print('ChromaDB OK')"
python -c "import neo4j; print('Neo4j OK')"
```

All four should print their `OK` message with no errors.

---

### Configuration

**Step 1 — Create your `.env` file** in the project root:

```bash
# .env — paste your Neo4j AuraDB credentials here
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-generated-password
```

**Step 2 — Review `config.yaml`** (defaults work for most use cases):

```yaml
paths:
  data_dir: "data/raw"          # where you drop your PDFs
  chroma_dir: "data/chroma_db"  # local vector store (auto-created)

ollama:
  base_url: "http://localhost:11434"
  llm_model: "llama3"
  embed_model: "nomic-embed-text"
  request_timeout: 300.0         # increase if your machine is slow

ingestion:
  chunk_size: 384                # tokens per chunk — good for lecture prose
  chunk_overlap: 64
  kg_max_triplets_per_chunk: 10

neo4j:
  database: "neo4j"
```

---

## Usage

### Step 1 — Add Your Source Material

Place your lecture PDFs or `.txt` files into `data/raw/`:

```
data/raw/
├── Lecture_01_Introduction.pdf
├── Lecture_02_Core_Concepts.pdf
└── Lecture_03_Advanced_Topics.pdf
```

File naming matters — the filename becomes the source citation shown in 
the UI. Use descriptive names like `Lecture_03_Cold_War.pdf` rather than 
`scan001.pdf`.

> **Supported formats:** Text-based PDFs (lecture slides, notes, papers) and 
> plain `.txt` files. Scanned PDFs (image-only) are not supported without 
> a separate OCR preprocessing step.

---

### Step 2 — Run Ingestion

```bash
python -m src.ingestion.pipeline
```

This will:
1. Parse all PDFs using PyMuPDF's layout-preserving block extractor
2. Split text into overlapping chunks using SentenceSplitter
3. Extract knowledge graph triplets via Llama-3 (the slow step — ~15-30s per chunk)
4. Write entity-relationship nodes to Neo4j AuraDB
5. Generate and write embeddings to local ChromaDB

**Typical time:** ~5-10 minutes for a 50-page PDF on M-series Mac.

**Incremental mode** — adding new lecture files later:

```bash
# Automatically detects and ingests only new files
python -m src.ingestion.pipeline

# Force complete rebuild of both databases
python -m src.ingestion.pipeline --force
```

After ingestion, verify your knowledge graph in the 
[Neo4j AuraDB browser](https://console.neo4j.io) by running:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
```

You should see a visual graph of entities extracted from your lectures.

---

### Step 3 — Launch the UI

```bash
# Ensure Ollama is still running (ollama serve in another terminal)
streamlit run app.py
```

Navigate to `http://localhost:8501`. The UI will:
- Connect to Neo4j and ChromaDB on startup (cached — fast after first load)
- Accept natural language questions about your course material
- Display the agent's step-by-step reasoning in the left sidebar
- Show source attribution chips (which lecture each answer came from)
- Show the three-metric quality scorecard (Groundedness / Relevance / Completeness)
- Flag answers that were automatically corrected before display

**Example queries to try:**

```
# Factual
"When did [event] happen and who were the key figures involved?"

# Relational (exercises the graph path)
"How does [concept A] relate to [concept B]?"
"What are the causal steps from [cause] to [effect]?"

# Conceptual
"Explain [definition] in your own words."
"What are the key properties of [concept]?"
```

---

### Step 4 — Run the Evaluation Suite

First, populate `data/eval/question_bank.json` with questions from your 
actual course material. The file ships with 5 example questions — replace 
the placeholder entries (`"REPLACE THIS"`) with domain-specific questions 
and their ground-truth answers.

```bash
# Quick smoke test — first 2 questions (~8 minutes)
python run_eval.py --limit 2

# Run one category
python run_eval.py --category factual

# Full suite (all non-placeholder questions)
python run_eval.py

# Run specific question IDs
python run_eval.py --id q001 q003 q005
```

Results are saved to `data/eval/results/eval_YYYYMMDD_HHMMSS.csv`.

---

## Evaluation

The evaluation suite measures four metrics with mathematical definitions 
equivalent to the [Ragas](https://docs.ragas.io) benchmark framework, 
reimplemented using local Ollama to remove the OpenAI dependency.

| Metric | Definition | Range |
|---|---|---|
| **Faithfulness** | `\|supported claims\| / \|total answer claims\|` | 0 → 1 |
| **Answer Relevancy** | Mean cosine similarity between reverse-generated questions and original query | 0 → 1 |
| **Context Precision** | Weighted precision rewarding relevant chunks at higher ranks | 0 → 1 |
| **Context Recall** | `\|GT claims covered by context\| / \|GT claims\|` | 0 → 1 |

**Sample output:**

```
═════════════════════════════════════════════════════════════════
   AGENTIC GRAPHRAG v2.0 — EVALUATION REPORT
═════════════════════════════════════════════════════════════════
   Pass Rate  : 16/20 (80.0%)  [threshold ≥ 0.65]
─────────────────────────────────────────────────────────────────
  ✅ Faithfulness            0.8750  [█████████████████░░░]
  ✅ Answer Relevancy        0.8123  [████████████████░░░░]
  ⚠️  Context Precision      0.6800  [█████████████░░░░░░░]
  ✅ Context Recall          0.7600  [███████████████░░░░░]
  ─────────────────────────────────────────────────────────────
  🏆 OVERALL (mean)          0.7568  [███████████████░░░░░]
═════════════════════════════════════════════════════════════════
```

---

## How the RRF Algorithm Works

Standard RAG systems face a fundamental incompatibility problem: vector 
databases return cosine similarity scores (continuous, range 0–1) while 
graph traversal returns nodes ordered by hop distance (discrete). These 
cannot be meaningfully combined by averaging or concatenation.

Reciprocal Rank Fusion solves this by discarding raw scores entirely 
and operating only on rank position:

```
RRF(d) = Σ   1 / (k + rank_R(d))
       R ∈ {R_vector, R_graph}
```

Where `k = 60` is a smoothing constant that dampens the influence of 
rank-1 results. A document appearing at rank 3 in vector search and 
rank 4 in graph traversal receives additive score contributions from 
both, naturally surfacing consensus results — content that both retrieval 
systems independently considered relevant.

---

## Project Structure

```
agentic-graphrag/
├── data/eval/question_bank.json   # Evaluation questions + ground truths
├── src/ingestion/pipeline.py      # Async batch ingestion with semaphore gating
├── src/ingestion/pdf_parser.py    # PyMuPDF layout-aware extraction
├── src/ingestion/chunker.py       # SentenceSplitter with metadata injection
├── src/retrieval/rrf_fusion.py    # Reciprocal Rank Fusion engine
├── src/retrieval/vector_retriever.py  # Async ChromaDB retriever
├── src/retrieval/graph_retriever.py   # Async Neo4j PropertyGraph retriever
├── src/agent/workflow.py          # End-to-end async agent pipeline
├── src/agent/self_correct.py      # 3-metric self-correction module
├── src/eval/evaluator.py          # Ragas-equivalent metric implementations
├── src/eval/report.py             # CSV + terminal report generator
├── app.py                         # Streamlit UI
└── run_eval.py                    # Evaluation suite CLI
```

---

## Acknowledgements

- **Prof. Shyamanta M. Hazarika**, IIT Guwahati — for project guidance
- [LlamaIndex](https://llamaindex.ai) — PropertyGraph and retrieval orchestration
- [Ollama](https://ollama.com) — local LLM inference with Metal acceleration
- [Ragas](https://docs.ragas.io) — evaluation metric definitions
- Cormack, Clarke & Buettcher (2009) — *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*

---

## License

MIT License — see [LICENSE](LICENSE) for details.
