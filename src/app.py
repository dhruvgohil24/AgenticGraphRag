# app.py

"""
Agentic GraphRAG — Streamlit Frontend

Layout:
┌─────────────────┬──────────────────────────────────────┐
│   SIDEBAR       │           MAIN PANEL                 │
│                 │                                      │
│  Agent Thought  │   Chat history (user + assistant)    │
│  Process Log    │                                      │
│                 │   [Route badge] [Verdict badge]      │
│  - Step 1       │                                      │
│  - Step 2       │   ┌──────────────────────────────┐  │
│  - Step 3       │   │ Type your question here...   │  │
│  - Step 4       │   └──────────────────────────────┘  │
└─────────────────┴──────────────────────────────────────┘
"""

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from src.utils import load_config, get_logger
from src.graph_store import load_graph_index, load_vector_index
from src.router import route_query
from src.self_correct import self_correct

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic GraphRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — makes it look like a real product, not a default Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3250;
    }

    /* Chat message bubbles */
    [data-testid="stChatMessage"] {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 4px 8px;
        margin-bottom: 8px;
        border: 1px solid #2e3250;
    }

    /* Route badge styles */
    .badge-vector {
        background-color: #1a3a5c;
        color: #4da6ff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid #4da6ff;
    }
    .badge-graph {
        background-color: #1a3d2e;
        color: #4dffaa;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid #4dffaa;
    }
    .badge-approved {
        background-color: #1a3d2e;
        color: #4dffaa;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #4dffaa;
    }
    .badge-corrected {
        background-color: #3d2e1a;
        color: #ffaa4d;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #ffaa4d;
    }
    .badge-insufficient {
        background-color: #3d1a1a;
        color: #ff6b6b;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #ff6b6b;
    }

    /* Thought log entry */
    .thought-entry {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #8892b0;
        padding: 2px 0;
        border-left: 2px solid #2e3250;
        padding-left: 8px;
        margin: 3px 0;
    }

    /* Score bar label */
    .score-label {
        font-size: 13px;
        color: #8892b0;
    }

    /* Header */
    .main-header {
        font-size: 28px;
        font-weight: 700;
        color: #ccd6f6;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 14px;
        color: #8892b0;
        margin-top: -10px;
    }

    /* Input box */
    [data-testid="stChatInput"] {
        background-color: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 12px;
    }

    /* Divider */
    hr { border-color: #2e3250; }
</style>
""", unsafe_allow_html=True)

def render_answer(text: str):
    """
    Render answer with proper Markdown and LaTeX support.
    Streamlit's st.markdown renders $...$ and $$...$$ natively.
    """
    st.markdown(text, unsafe_allow_html=False)
# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []            # chat history
    if "thought_logs" not in st.session_state:
        st.session_state.thought_logs = []        # per-message thought logs
    if "graph_index" not in st.session_state:
        st.session_state.graph_index = None
    if "vector_index" not in st.session_state:
        st.session_state.vector_index = None
    if "indexes_loaded" not in st.session_state:
        st.session_state.indexes_loaded = False
    if "config" not in st.session_state:
        st.session_state.config = None


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADER
# Streamlit re-runs the whole script on every interaction.
# @st.cache_resource ensures the heavy index loading only happens ONCE.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all_indexes():
    """Load both indexes once and cache them for the entire session."""
    load_dotenv()
    config = load_config()

    Settings.llm = Ollama(
        model=config["ollama"]["llm_model"],
        base_url=config["ollama"]["base_url"],
        request_timeout=config["ollama"]["request_timeout"],
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config["ollama"]["embed_model"],
        base_url=config["ollama"]["base_url"],
    )

    graph_index  = load_graph_index(config)
    vector_index = load_vector_index(config)
    return config, graph_index, vector_index


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Agent Thought Process
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(thought_logs: list):
    with st.sidebar:
        st.markdown("### 🤖 Agent Thought Process")
        st.markdown("---")

        if not thought_logs:
            st.markdown(
                "<p style='color:#8892b0; font-size:13px;'>"
                "Ask a question to see the agent's<br>reasoning steps here."
                "</p>",
                unsafe_allow_html=True,
            )
            return

        # Show the most recent thought log at the top
        latest = thought_logs[-1]
        msg_num = len(thought_logs)

        st.markdown(
            f"<p style='color:#ccd6f6; font-size:13px; font-weight:600;'>"
            f"Latest query (#{msg_num})</p>",
            unsafe_allow_html=True,
        )

        for entry in latest["log"]:
            st.markdown(
                f"<div class='thought-entry'>{entry}</div>",
                unsafe_allow_html=True,
            )

        # Faithfulness score meter
        score = latest.get("faithfulness_score", 0)
        if score > 0:
            st.markdown("---")
            st.markdown(
                "<p class='score-label'>Faithfulness Score</p>",
                unsafe_allow_html=True,
            )
            score_color = (
                "#4dffaa" if score >= 7
                else "#ffaa4d" if score >= 4
                else "#ff6b6b"
            )
            st.markdown(
                f"<p style='font-size:24px; font-weight:700; "
                f"color:{score_color}; margin:0'>{score}/10</p>",
                unsafe_allow_html=True,
            )
            st.progress(score / 10)

        # Previous queries (collapsed)
        if len(thought_logs) > 1:
            st.markdown("---")
            with st.expander(f"Previous queries ({len(thought_logs)-1})"):
                for i, log_entry in enumerate(reversed(thought_logs[:-1])):
                    st.markdown(
                        f"<p style='color:#ccd6f6; font-size:12px; "
                        f"font-weight:600;'>Query #{len(thought_logs)-1-i}</p>",
                        unsafe_allow_html=True,
                    )
                    for entry in log_entry["log"]:
                        st.markdown(
                            f"<div class='thought-entry'>{entry}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# BADGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def route_badge(route: str) -> str:
    cls = "badge-vector" if route == "vector" else "badge-graph"
    icon = "📚" if route == "vector" else "🕸️"
    label = route.upper()
    return f"<span class='{cls}'>{icon} {label}</span>"

def verdict_badge(verdict: str) -> str:
    mapping = {
        "APPROVED": ("badge-approved", "✅"),
        "CORRECTED": ("badge-corrected", "⚠️"),
        "INSUFFICIENT_CONTEXT": ("badge-insufficient", "❓"),
    }
    cls, icon = mapping.get(verdict, ("badge-approved", "✅"))
    return f"<span class='{cls}'>{icon} {verdict}</span>"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_main(config, graph_index, vector_index):
    # Header
    st.markdown(
        "<p class='main-header'>🧠 Agentic GraphRAG</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='sub-header'>"
        "Educational Q&A with Knowledge Graph + Vector Search · "
        "Powered by Llama 3 (local) · Neo4j · ChromaDB"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Show badges above the answer
                badges = (
                    route_badge(msg.get("route", "vector"))
                    + "&nbsp;&nbsp;"
                    + verdict_badge(msg.get("verdict", "APPROVED"))
                    + f"&nbsp;&nbsp;<span style='color:#8892b0; font-size:12px;'>"
                    f"⏱ {msg.get('latency', '?')}s</span>"
                )
                st.markdown(badges, unsafe_allow_html=True)
                st.markdown(msg["content"])

                if msg.get("was_corrected"):
                    with st.expander("📋 View original (pre-correction) answer"):
                        st.markdown(
                            f"<p style='color:#8892b0; font-size:13px;'>"
                            f"{msg.get('original_answer', '')}</p>",
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(msg["content"])

    # Chat input
    if user_query := st.chat_input("Ask anything about your course materials..."):
        # Immediately show the user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
        })
        with st.chat_message("user"):
            render_answer(user_query)

        # Process the query with a loading spinner
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):

                # ── Router ────────────────────────────────────────────────
                router_resp = route_query(
                    query=user_query,
                    graph_index=graph_index,
                    vector_index=vector_index,
                    config=config,
                )

                # ── Self-Correction ───────────────────────────────────────
                correction = self_correct(
                    query=user_query,
                    answer=router_resp.answer,
                    retrieved_context=router_resp.retrieved_context,
                    route_taken=router_resp.route_taken,
                )

            # ── Render answer ─────────────────────────────────────────────
            badges = (
                route_badge(router_resp.route_taken)
                + "&nbsp;&nbsp;"
                + verdict_badge(correction.verdict)
                + f"&nbsp;&nbsp;<span style='color:#8892b0; font-size:12px;'>"
                f"⏱ {router_resp.latency_seconds}s</span>"
            )
            st.markdown(badges, unsafe_allow_html=True)
            render_answer(correction.final_answer)

            if correction.was_corrected:
                with st.expander("📋 View original (pre-correction) answer"):
                    st.markdown(
                        f"<p style='color:#8892b0; font-size:13px;'>"
                        f"{correction.original_answer}</p>",
                        unsafe_allow_html=True,
                    )

        # ── Persist to session state ──────────────────────────────────────
        st.session_state.messages.append({
            "role": "assistant",
            "content": correction.final_answer,
            "route": router_resp.route_taken,
            "verdict": correction.verdict,
            "latency": router_resp.latency_seconds,
            "was_corrected": correction.was_corrected,
            "original_answer": correction.original_answer,
        })

        # ── Store thought log for sidebar ─────────────────────────────────
        full_log = router_resp.thought_log + correction.correction_log
        st.session_state.thought_logs.append({
            "log": full_log,
            "faithfulness_score": correction.faithfulness_score,
        })

        # Force sidebar to re-render with latest log
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_session_state()

    # Load indexes (cached after first run)
    with st.spinner("🔗 Connecting to Neo4j and ChromaDB..."):
        try:
            config, graph_index, vector_index = load_all_indexes()
        except Exception as e:
            st.error(
                f"❌ Failed to load indexes: {e}\n\n"
                "Make sure:\n"
                "1. Ollama is running (`ollama serve`)\n"
                "2. Your `.env` file has correct Neo4j credentials\n"
                "3. You've run `python -m src.ingestion` at least once"
            )
            st.stop()

    render_sidebar(st.session_state.thought_logs)
    render_main(config, graph_index, vector_index)


if __name__ == "__main__":
    main()