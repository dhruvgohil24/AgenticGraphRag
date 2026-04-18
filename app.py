# app.py
"""
Agentic GraphRAG v2.0 — Streamlit Frontend

Layout:
┌──────────────────────┬─────────────────────────────────────────────────┐
│  SIDEBAR             │  MAIN PANEL                                     │
│                      │                                                  │
│  Agent Thought Log   │  Chat history                                   │
│  ─────────────────   │                                                  │
│  Metric Scorecard    │  [HYBRID-RRF] [APPROVED] [score] [latency]     │
│  ─────────────────   │                                                  │
│  Composite Score     │  Answer text                                    │
│  Progress bar        │                                                  │
│  ─────────────────   │  Sources: Lecture A [VECTOR], Lecture B [GRAPH] │
│  Previous queries    │                                                  │
│  (collapsed)         │  ┌──────────────────────────────────────────┐  │
│                      │  │ Ask anything about your course materials │  │
│                      │  └──────────────────────────────────────────┘  │
└──────────────────────┴─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import traceback
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from src.agent.workflow import AgentResponse, AgentWorkflow
from src.utils import get_logger, load_config

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the absolute first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAG v2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# All colours, spacing, and component styles defined once here.
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "bg_primary":    "#0d1117",
    "bg_secondary":  "#161b22",
    "bg_card":       "#1c2128",
    "border":        "#30363d",
    "text_primary":  "#e6edf3",
    "text_muted":    "#7d8590",
    "text_code":     "#79c0ff",
    "green":         "#3fb950",
    "green_bg":      "#1a2d1a",
    "orange":        "#d29922",
    "orange_bg":     "#2d2206",
    "blue":          "#58a6ff",
    "blue_bg":       "#0d1d2e",
    "red":           "#f85149",
    "red_bg":        "#2d1515",
    "purple":        "#bc8cff",
    "purple_bg":     "#1e1530",
}

st.markdown(f"""
<style>
    /* ── Reset & Base ───────────────────────────────────────────── */
    .stApp {{
        background-color: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
    }}
    .stApp > header {{
        background-color: transparent;
    }}

    /* ── Sidebar ────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_secondary']};
        border-right: 1px solid {COLORS['border']};
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {COLORS['text_muted']};
        font-size: 13px;
        line-height: 1.6;
    }}

    /* ── Chat messages ──────────────────────────────────────────── */
    [data-testid="stChatMessage"] {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 6px 10px;
        margin-bottom: 10px;
    }}

    /* ── Chat input ─────────────────────────────────────────────── */
    [data-testid="stChatInput"] textarea {{
        background-color: {COLORS['bg_secondary']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['text_primary']} !important;
        border-radius: 10px !important;
    }}

    /* ── Expander ───────────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        background-color: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}

    /* ── Thought log entry ──────────────────────────────────────── */
    .thought-line {{
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 11.5px;
        color: {COLORS['text_muted']};
        padding: 2px 0 2px 10px;
        border-left: 2px solid {COLORS['border']};
        margin: 2px 0;
        line-height: 1.5;
        word-break: break-word;
    }}
    .thought-line.highlight {{
        color: {COLORS['text_code']};
        border-left-color: {COLORS['blue']};
    }}

    /* ── Metric card ────────────────────────────────────────────── */
    .metric-card {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 10px 12px;
        margin: 4px 0;
    }}
    .metric-name {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {COLORS['text_muted']};
        margin-bottom: 4px;
    }}
    .metric-score-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }}
    .metric-score-value {{
        font-size: 22px;
        font-weight: 700;
        line-height: 1;
    }}
    .metric-score-denom {{
        font-size: 13px;
        color: {COLORS['text_muted']};
    }}
    .metric-weight {{
        font-size: 11px;
        color: {COLORS['text_muted']};
        background: {COLORS['bg_secondary']};
        padding: 2px 6px;
        border-radius: 10px;
        border: 1px solid {COLORS['border']};
    }}
    .metric-reason {{
        font-size: 11.5px;
        color: {COLORS['text_muted']};
        font-style: italic;
        line-height: 1.4;
        margin-top: 4px;
    }}

    /* ── Composite score ────────────────────────────────────────── */
    .composite-label {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {COLORS['text_muted']};
        margin-top: 12px;
        margin-bottom: 6px;
    }}
    .composite-value {{
        font-size: 36px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -1px;
    }}

    /* ── Badge row ──────────────────────────────────────────────── */
    .badge-row {{
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }}
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid;
    }}
    .badge-hybrid {{
        background: {COLORS['purple_bg']};
        color: {COLORS['purple']};
        border-color: {COLORS['purple']};
    }}
    .badge-approved {{
        background: {COLORS['green_bg']};
        color: {COLORS['green']};
        border-color: {COLORS['green']};
    }}
    .badge-corrected {{
        background: {COLORS['orange_bg']};
        color: {COLORS['orange']};
        border-color: {COLORS['orange']};
    }}
    .badge-requery {{
        background: {COLORS['blue_bg']};
        color: {COLORS['blue']};
        border-color: {COLORS['blue']};
    }}
    .badge-error, .badge-insufficient {{
        background: {COLORS['red_bg']};
        color: {COLORS['red']};
        border-color: {COLORS['red']};
    }}
    .badge-meta {{
        background: transparent;
        color: {COLORS['text_muted']};
        border-color: {COLORS['border']};
    }}
    .badge-attempt {{
        background: {COLORS['orange_bg']};
        color: {COLORS['orange']};
        border-color: {COLORS['orange']};
    }}

    /* ── Source chips ───────────────────────────────────────────── */
    .sources-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid {COLORS['border']};
    }}
    .sources-label {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: {COLORS['text_muted']};
        width: 100%;
        margin-bottom: 2px;
    }}
    .source-chip-vector {{
        background: {COLORS['blue_bg']};
        color: {COLORS['blue']};
        border: 1px solid {COLORS['blue']};
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 500;
    }}
    .source-chip-graph {{
        background: {COLORS['green_bg']};
        color: {COLORS['green']};
        border: 1px solid {COLORS['green']};
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 500;
    }}

    /* ── Header ─────────────────────────────────────────────────── */
    .app-title {{
        font-size: 26px;
        font-weight: 800;
        color: {COLORS['text_primary']};
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }}
    .app-subtitle {{
        font-size: 13px;
        color: {COLORS['text_muted']};
    }}

    /* ── Divider ─────────────────────────────────────────────────── */
    hr {{
        border: none;
        border-top: 1px solid {COLORS['border']};
        margin: 12px 0;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE SCHEMA
# Defined once here. Any key not in this schema will raise a KeyError later.
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    defaults: dict = {
        # List of message dicts — full chat history
        "messages": [],
        # Parallel list of sidebar data dicts — one per assistant message
        "sidebar_data": [],
        # Tracks whether the system is currently processing a query
        "is_processing": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# Message dict schema (stored in st.session_state.messages):
# {
#   "role": "user" | "assistant",
#   "content": str,
#   # assistant-only fields:
#   "verdict": str,
#   "composite_score": float,
#   "was_corrected": bool,
#   "original_answer": str,
#   "attempt_count": int,
#   "latency_seconds": float,
#   "source_attributions": list[str],
#   "metric_scores": dict,
#   "error": str | None,
# }


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE: AgentWorkflow
# Instantiated ONCE per Streamlit process, reused across all interactions.
# @st.cache_resource persists across reruns; it will NOT be re-created
# when the user types a new message.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_workflow() -> tuple[AgentWorkflow, Optional[str]]:
    """
    Load config, configure LlamaIndex Settings, and instantiate AgentWorkflow.
    Returns (workflow, error_message). error_message is None on success.
    """
    try:
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

        workflow = AgentWorkflow(config)
        logger.info("✅ AgentWorkflow loaded and cached.")
        return workflow, None

    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(f"Failed to load workflow: {error}")
        return None, error  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC RUNNER
# Streamlit is synchronous. AgentWorkflow.run() is async.
# We solve this by spawning a fresh OS thread that owns its own event loop.
# This is the only safe pattern — asyncio.run() would fail if Streamlit
# has already set up a loop in the main thread.
# ─────────────────────────────────────────────────────────────────────────────

def _run_workflow_sync(workflow: AgentWorkflow, query: str) -> AgentResponse:
    """
    Bridge between Streamlit's sync context and the async workflow.

    Creates a brand-new event loop in a dedicated thread so:
    1. No conflict with any loop Streamlit may have running
    2. The async workflow can itself spawn sub-tasks freely
    3. ThreadPoolExecutor calls inside the workflow get a clean loop

    This function is called via st.spinner() — it blocks until complete.
    """
    def _thread_target() -> AgentResponse:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(workflow.run(query))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_thread_target)
        return future.result(timeout=600)   # 10-minute hard timeout


# ─────────────────────────────────────────────────────────────────────────────
# BADGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_badge(verdict: str) -> str:
    mapping = {
        "APPROVED":             ("badge-approved",    "✅ APPROVED"),
        "CORRECTED":            ("badge-corrected",   "⚠️ CORRECTED"),
        "REQUERY":              ("badge-requery",     "🔄 REQUERIED"),
        "INSUFFICIENT_CONTEXT": ("badge-insufficient","❓ LOW CONTEXT"),
        "ERROR":                ("badge-error",       "❌ ERROR"),
    }
    cls, label = mapping.get(verdict, ("badge-approved", verdict))
    return f"<span class='badge {cls}'>{label}</span>"


def _route_badge() -> str:
    return "<span class='badge badge-hybrid'>⚡ HYBRID-RRF</span>"


def _latency_badge(seconds: float) -> str:
    return f"<span class='badge badge-meta'>⏱ {seconds}s</span>"


def _score_badge(score: float) -> str:
    color_cls = (
        "badge-approved"    if score >= 7.5 else
        "badge-corrected"   if score >= 5.0 else
        "badge-insufficient"
    )
    return f"<span class='badge {color_cls}'>📊 {score:.1f}/10</span>"


def _attempt_badge(attempts: int) -> str:
    if attempts == 1:
        return ""
    return f"<span class='badge badge-attempt'>🔄 {attempts} attempts</span>"


# ─────────────────────────────────────────────────────────────────────────────
# METRIC SCORECARD RENDERER
# Renders the 3-metric DSPy scorecard in the sidebar.
# ─────────────────────────────────────────────────────────────────────────────

def _render_metric_card(
    name: str,
    data: dict,
    weight: float,
) -> None:
    """
    Render one metric as a styled card in the sidebar.

    Args:
        name:   metric key e.g. "groundedness"
        data:   {"score": float, "reasoning": str, "passed": bool}
        weight: 0.0-1.0 weight for this metric
    """
    score = data.get("score", 0.0)
    reasoning = data.get("reasoning", "")
    passed = data.get("passed", False)

    # Color based on score
    if score >= 8:
        score_color = COLORS["green"]
    elif score >= 6:
        score_color = COLORS["orange"]
    else:
        score_color = COLORS["red"]

    status_icon = "✅" if passed else "❌"
    display_name = name.replace("_", " ").title()
    weight_pct = f"{int(weight * 100)}%"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-name">{status_icon} {display_name}</div>
        <div class="metric-score-row">
            <span class="metric-score-value" style="color:{score_color}">
                {score:.1f}
            </span>
            <span class="metric-score-denom">/10</span>
            <span class="metric-weight">weight: {weight_pct}</span>
        </div>
        <div class="metric-reason">{reasoning}</div>
    </div>
    """, unsafe_allow_html=True)


_METRIC_WEIGHTS = {
    "groundedness":       0.50,
    "context_relevance":  0.30,
    "answer_completeness": 0.20,
}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<p style='font-size:16px; font-weight:700; "
            f"color:{COLORS['text_primary']};'>🤖 Agent Thought Process</p>",
            unsafe_allow_html=True,
        )

        sidebar_data_list: list[dict] = st.session_state.sidebar_data

        # Empty state
        if not sidebar_data_list:
            st.markdown(
                f"<p style='color:{COLORS['text_muted']}; font-size:13px;'>"
                "Ask a question to see the agent's<br>"
                "step-by-step reasoning here."
                "</p>",
                unsafe_allow_html=True,
            )
            return

        # Latest query
        latest = sidebar_data_list[-1]
        q_num = len(sidebar_data_list)

        st.markdown(
            f"<p style='font-size:12px; font-weight:600; "
            f"color:{COLORS['text_primary']};'>"
            f"Latest query (#{q_num})</p>",
            unsafe_allow_html=True,
        )

        # Thought log
        thought_log: list[str] = latest.get("thought_log", [])
        for line in thought_log:
            # Highlight key lines
            is_highlight = any(
                marker in line for marker in ["Step", "✅", "❌", "🏁", "⚡", "🔀"]
            )
            css_class = "thought-line highlight" if is_highlight else "thought-line"
            # Escape HTML special chars in the line
            safe_line = (
                line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            st.markdown(
                f"<div class='{css_class}'>{safe_line}</div>",
                unsafe_allow_html=True,
            )

        # Metric scorecard
        metric_scores: dict = latest.get("metric_scores", {})
        composite: float = latest.get("composite_score", 0.0)

        if metric_scores:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='font-size:12px; font-weight:600; "
                f"color:{COLORS['text_primary']};'>📊 Quality Metrics</p>",
                unsafe_allow_html=True,
            )

            for metric_name, data in metric_scores.items():
                weight = _METRIC_WEIGHTS.get(metric_name, 0.0)
                _render_metric_card(metric_name, data, weight)

            # Composite score
            if composite >= 7.5:
                composite_color = COLORS["green"]
            elif composite >= 5.0:
                composite_color = COLORS["orange"]
            else:
                composite_color = COLORS["red"]

            st.markdown(
                f"<div class='composite-label'>Composite Score</div>"
                f"<div class='composite-value' "
                f"style='color:{composite_color}'>"
                f"{composite:.2f}"
                f"<span style='font-size:16px; color:{COLORS['text_muted']}; "
                f"font-weight:400'> / 10</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(composite / 10.0, 1.0))

        # Previous queries (collapsed)
        if len(sidebar_data_list) > 1:
            st.markdown("<hr>", unsafe_allow_html=True)
            with st.expander(
                f"📋 Previous queries ({len(sidebar_data_list) - 1})"
            ):
                for i, past in enumerate(reversed(sidebar_data_list[:-1])):
                    past_num = len(sidebar_data_list) - 1 - i
                    score = past.get("composite_score", 0.0)
                    verdict = past.get("verdict", "")
                    st.markdown(
                        f"<p style='font-size:11px; font-weight:600; "
                        f"color:{COLORS['text_primary']};'>"
                        f"Query #{past_num} — {verdict} — {score:.1f}/10"
                        f"</p>",
                        unsafe_allow_html=True,
                    )
                    for line in past.get("thought_log", [])[:5]:
                        safe = (
                            line.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                        )
                        st.markdown(
                            f"<div class='thought-line'>{safe}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE ATTRIBUTION RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_source_chips(attributions: list[str]) -> None:
    """
    Render source lecture chips below an answer.
    Each chip is colour-coded by source type (VECTOR=blue, GRAPH=green).
    """
    if not attributions:
        return

    chips_html = "<div class='sources-row'>"
    chips_html += "<span class='sources-label'>📚 Sources</span>"

    for attr in attributions:
        if "[VECTOR]" in attr:
            lecture = attr.replace(" [VECTOR]", "")
            chips_html += (
                f"<span class='source-chip-vector'>"
                f"📄 {lecture}"
                f"</span>"
            )
        elif "[GRAPH]" in attr:
            lecture = attr.replace(" [GRAPH]", "")
            chips_html += (
                f"<span class='source-chip-graph'>"
                f"🕸️ {lecture}"
                f"</span>"
            )
        else:
            chips_html += (
                f"<span class='source-chip-vector'>{attr}</span>"
            )

    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ASSISTANT MESSAGE RENDERER
# Encapsulates all the logic for rendering one assistant turn.
# Called both when replaying history and when rendering a new response.
# ─────────────────────────────────────────────────────────────────────────────

def _render_assistant_message(msg: dict) -> None:
    """
    Render one assistant message with all its metadata.

    Args:
        msg: Message dict from st.session_state.messages
    """
    # Badge row
    badges = (
        _route_badge()
        + " "
        + _verdict_badge(msg.get("verdict", "APPROVED"))
        + " "
        + _score_badge(msg.get("composite_score", 0.0))
        + " "
        + _latency_badge(msg.get("latency_seconds", 0.0))
        + " "
        + _attempt_badge(msg.get("attempt_count", 1))
    )
    st.markdown(
        f"<div class='badge-row'>{badges}</div>",
        unsafe_allow_html=True,
    )

    # Main answer
    st.markdown(msg["content"])

    # Source chips
    _render_source_chips(msg.get("source_attributions", []))

    # Corrected answer expander
    if msg.get("was_corrected") and msg.get("original_answer"):
        with st.expander("📋 View original answer (before correction)"):
            st.markdown(
                f"<p style='color:{COLORS['text_muted']}; font-size:13px;'>"
                f"{msg['original_answer']}"
                f"</p>",
                unsafe_allow_html=True,
            )

    # Error expander
    if msg.get("error"):
        with st.expander("🔴 Error details"):
            st.code(msg["error"], language="python")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_main(workflow: AgentWorkflow) -> None:
    """
    Render the header, chat history, and input box.
    Handles new user input via the workflow pipeline.
    """

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='app-title'>🧠 Agentic GraphRAG <span style='font-size:14px; "
        "font-weight:400; color:#7d8590'>v2.0</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-subtitle'>"
        "Hybrid RRF Retrieval · 3-Metric Self-Correction · "
        "Llama-3 (local) · Neo4j · ChromaDB"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant_message(msg)

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input: Optional[str] = st.chat_input(
        "Ask anything about your course materials...",
        disabled=st.session_state.is_processing,
    )

    if not user_input:
        return

    # Immediately render user bubble
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Process query ─────────────────────────────────────────────────────────
    st.session_state.is_processing = True

    with st.chat_message("assistant"):
        with st.spinner("⚡ Agent running — retrieval + synthesis + verification..."):
            try:
                response: AgentResponse = _run_workflow_sync(
                    workflow, user_input
                )
            except Exception as e:
                error_str = traceback.format_exc()
                logger.error(f"Workflow execution error: {error_str}")
                response = AgentResponse(
                    query=user_input,
                    final_answer=(
                        "A critical error occurred during processing. "
                        "Please check the terminal logs."
                    ),
                    verdict="ERROR",
                    error=error_str,
                )

        # Render the new response immediately
        msg_dict: dict = {
            "role": "assistant",
            "content": response.final_answer,
            "verdict": response.verdict,
            "composite_score": response.composite_score,
            "was_corrected": response.was_corrected,
            "original_answer": response.original_answer
                if response.was_corrected else "",
            "attempt_count": response.attempt_count,
            "latency_seconds": response.latency_seconds,
            "source_attributions": response.source_attributions,
            "metric_scores": response.metric_scores,
            "error": response.error,
        }
        _render_assistant_message(msg_dict)

    # ── Persist to session state ──────────────────────────────────────────────
    st.session_state.messages.append(msg_dict)

    st.session_state.sidebar_data.append({
        "thought_log":     response.thought_log,
        "metric_scores":   response.metric_scores,
        "composite_score": response.composite_score,
        "verdict":         response.verdict,
    })

    st.session_state.is_processing = False

    # Force full rerun so sidebar updates with the latest thought log
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    # Load workflow (cached after first call)
    with st.spinner("🔗 Connecting to Neo4j, ChromaDB, and Ollama..."):
        workflow, load_error = _load_workflow()

    if load_error or workflow is None:
        st.error("❌ Failed to initialize the AgentWorkflow.")
        st.markdown("**Checklist:**")
        st.markdown(
            "- Ollama is running: `ollama serve`\n"
            "- `.env` contains valid `NEO4J_URI`, `NEO4J_USERNAME`, "
            "`NEO4J_PASSWORD`\n"
            "- Ingestion has been run: "
            "`python -m src.ingestion.pipeline`\n"
            "- Virtual environment is active with all dependencies"
        )
        if load_error:
            with st.expander("🔴 Full error traceback"):
                st.code(load_error, language="python")
        st.stop()

    _render_sidebar()
    _render_main(workflow)


if __name__ == "__main__":
    main()
