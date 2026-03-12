"""LLM request / response inspector page."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

from coach.database import (
    LlmCallRecord,
    get_llm_calls,
    get_llm_calls_stats,
    get_all_sessions_stats,
    init_db,
)
from coach.ui.utils import hide_streamlit_chrome

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Log — Productivity Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_chrome()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

init_db()

all_sessions = get_all_sessions_stats()  # newest first

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Productivity Coach")
    st.caption("LLM Log")
    st.divider()

    # Session filter
    session_options = {"All sessions": None}
    for s in all_sessions:
        label = s["start_time"].strftime("%b %d %H:%M")
        if s.get("goal"):
            label += f" — {s['goal'][:30]}"
        session_options[f"#{s['id']}  {label}"] = s["id"]

    selected_label = st.selectbox(
        "Filter by session",
        options=list(session_options.keys()),
    )
    selected_session_id: Optional[int] = session_options[selected_label]

    st.divider()

    # Call type filter
    call_type_options = [
        "All types",
        "analyse",
        "session_summary",
        "weekly_summary",
        "health_check",
    ]
    selected_type = st.selectbox("Call type", options=call_type_options)

    st.divider()
    st.page_link("Coach.py", label="Back to Coach", icon="🎯")

# ---------------------------------------------------------------------------
# Load records
# ---------------------------------------------------------------------------

records = get_llm_calls(session_id=selected_session_id, limit=500)

if selected_type != "All types":
    records = [r for r in records if r.call_type == selected_type]

stats = get_llm_calls_stats(session_id=selected_session_id)

# ---------------------------------------------------------------------------
# Header + summary stats
# ---------------------------------------------------------------------------

st.title("LLM Log")

if not records:
    st.info(
        "No LLM calls recorded yet. "
        "Calls are logged automatically from the next monitoring session onwards."
    )
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Calls", stats["total_calls"])
col2.metric(
    "Input Tokens", f"{stats['total_input']:,}" if stats["total_input"] else "—"
)
col3.metric(
    "Output Tokens", f"{stats['total_output']:,}" if stats["total_output"] else "—"
)
col4.metric(
    "Avg Latency",
    f"{int(stats['avg_latency_ms'])} ms" if stats["avg_latency_ms"] else "—",
)
col5.metric(
    "Errors",
    stats["error_count"],
    delta=None,
    help="Calls that returned an error",
)

# Shown subset vs total
if selected_type != "All types" or selected_session_id is not None:
    st.caption(
        f"Showing {len(records)} call{'s' if len(records) != 1 else ''} matching current filters."
    )

st.divider()

# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------

_TYPE_COLOURS = {
    "analyse": ("#1a6bbd", "#cce0f7"),
    "session_summary": ("#1a7a4a", "#c6f0d9"),
    "weekly_summary": ("#7a4a1a", "#f0ddc6"),
    "health_check": ("#555", "#e8e8e8"),
}

_TYPE_LABELS = {
    "analyse": "Analyse",
    "session_summary": "Session Summary",
    "weekly_summary": "Weekly Summary",
    "health_check": "Health Check",
}


def _badge(call_type: str) -> str:
    fg, bg = _TYPE_COLOURS.get(call_type, ("#555", "#e8e8e8"))
    label = _TYPE_LABELS.get(call_type, call_type)
    return (
        f"<span style='background:{bg}; color:{fg}; font-size:0.75rem; "
        f"font-weight:700; padding:2px 9px; border-radius:12px; "
        f"letter-spacing:0.03em;'>{label}</span>"
    )


def _status_icon(rec: LlmCallRecord) -> str:
    return "🔴" if rec.error else "🟢"


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(tz=None).strftime("%b %d  %H:%M:%S")


def _tokens_str(rec: LlmCallRecord) -> str:
    if rec.token_input is not None or rec.token_output is not None:
        i = rec.token_input or 0
        o = rec.token_output or 0
        return f"{i:,} in / {o:,} out"
    return "—"


def _latency_str(rec: LlmCallRecord) -> str:
    if rec.latency_ms is not None:
        if rec.latency_ms >= 1000:
            return f"{rec.latency_ms / 1000:.1f} s"
        return f"{rec.latency_ms} ms"
    return "—"


for rec in records:
    session_tag = f"Session #{rec.session_id}" if rec.session_id else "no session"
    expander_label = (
        f"{_status_icon(rec)}  {_fmt_ts(rec.timestamp)}  ·  "
        f"{_TYPE_LABELS.get(rec.call_type, rec.call_type)}  ·  "
        f"{session_tag}  ·  {_latency_str(rec)}"
    )

    with st.expander(expander_label, expanded=False):
        # Meta row
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns([2, 2, 2, 2])
        with meta_col1:
            st.markdown(_badge(rec.call_type), unsafe_allow_html=True)
        meta_col2.markdown(f"**Model:** `{rec.model}`")
        meta_col3.markdown(f"**Tokens:** {_tokens_str(rec)}")
        meta_col4.markdown(f"**Latency:** {_latency_str(rec)}")

        if rec.error:
            st.error(f"**Error:** {rec.error}")

        resp_tab, req_tab = st.tabs(["Response", "Request"])

        with resp_tab:
            if rec.response_text:
                # Try to pretty-print JSON responses
                try:
                    parsed = json.loads(rec.response_text)
                    st.json(parsed)
                except (json.JSONDecodeError, TypeError):
                    st.code(rec.response_text, language="markdown")
            else:
                st.caption("No response recorded.")

        with req_tab:
            st.code(rec.request_text, language="markdown")
