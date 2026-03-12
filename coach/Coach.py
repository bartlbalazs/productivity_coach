"""Streamlit UI for the Productivity Coach application.

This module is the thin page-layout orchestrator.  Business logic lives in
``session_controller``, state initialisation in ``session_state``, reusable
render helpers in ``ui_components``, and pure theme utilities in ``ui_theme``.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from typing import Optional

# Suppress FutureWarning from google-cloud-aiplatform about google-cloud-storage
# version compatibility — this is a library concern, not actionable by us.
warnings.filterwarnings(
    "ignore",
    message="Support for google-cloud-storage < 3.0.0 will be removed",
    category=FutureWarning,
    module="google.cloud.aiplatform",
)

import altair as alt
import pandas as pd
import streamlit as st

from coach.core.agent import UserMode
from coach.config import config
from coach.database import (
    SessionLogEntry,
    get_all_captures_for_session,
    get_or_cleanup_open_session,
    init_db,
)
from coach.integrations import tts
from coach.core.scheduler import MonitoringScheduler
from coach.core.session_controller import (
    auto_resume_if_needed,
    drain_event_queue,
    finish_task,
    resume_open_session,
    set_current_focus,
    start_monitoring,
    stop_monitoring,
    submit_log_entry,
)
from coach.core.session_state import init_state
from coach.ui.components import (
    render_capture_expander,
    render_coaching_card,
    render_errors,
    render_mode_banner,
    render_posture_callout,
    render_rest_banner,
    render_score_header,
    render_session_stats,
    render_sprint_timer,
    render_suggestion_cards,
    render_task_list,
)
from coach.ui.theme import distraction_badge, score_color, score_label, score_palette
from coach.ui.utils import hide_streamlit_chrome

# ---------------------------------------------------------------------------
# Logging — configure once for the whole process so all coach.* loggers
# emit to stdout with timestamps.  Streamlit's own loggers stay at WARNING.
# ---------------------------------------------------------------------------

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    # Keep third-party noise down — set once, persists for the process lifetime.
    for _noisy in (
        "streamlit",
        "urllib3",
        "httpx",
        "httpcore",
        "google",
        "grpc",
        "h2",
        "watchdog",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

# Streamlit loggers use propagate=False and attach their own StreamHandler.
# Setting level on the logger is overwritten when Streamlit re-initialises it
# via get_logger() → setLevel(_global_log_level).  A logging.Filter survives
# that reset because setLevel() does not touch the filters list.
# We attach the filter on every script run; the isinstance guard prevents
# duplicate filter instances from accumulating.


class _FragmentWarningFilter(logging.Filter):
    """Drop the noisy 'fragment does not exist anymore' INFO message."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return "does not exist anymore" not in record.getMessage()


_app_session_logger = logging.getLogger("streamlit.runtime.app_session")
if not any(isinstance(f, _FragmentWarningFilter) for f in _app_session_logger.filters):
    _app_session_logger.addFilter(_FragmentWarningFilter())

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Productivity Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit chrome: running-man activity indicator and deploy button.
hide_streamlit_chrome()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Score thresholds used for colour coding across the UI.
_SCORE_HIGH = 8
_SCORE_MID = 5


# ---------------------------------------------------------------------------
# Session log — current focus + free-form log entries
# ---------------------------------------------------------------------------


def _render_session_log() -> None:
    """Render the session log area at the top of the main page.

    Shows the current focus prominently and lets the user append free-form
    notes at any time. Each note is persisted via ``submit_log_entry`` and
    forwarded to the scheduler as the new session goal.
    """
    current_focus: str = st.session_state.get("current_focus", "")
    log_entries: list[SessionLogEntry] = st.session_state.get("session_log", [])

    # --- Current focus display ---
    if current_focus:
        st.markdown(
            f"<div style='background:rgba(74,144,217,0.12); border-left:4px solid #4a90d9; "
            f"border-radius:6px; padding:0.7rem 1.1rem; margin-bottom:0.8rem;'>"
            f"<span style='font-size:0.75rem; color:#888; text-transform:uppercase; "
            f"letter-spacing:0.05em;'>Current focus</span><br>"
            f"<span style='font-size:1.05rem; font-weight:600; color:#e8e8e8;'>"
            f"{current_focus}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:rgba(255,255,255,0.04); border-left:4px solid #555; "
            "border-radius:6px; padding:0.7rem 1.1rem; margin-bottom:0.8rem;'>"
            "<span style='font-size:0.75rem; color:#888; text-transform:uppercase; "
            "letter-spacing:0.05em;'>Current focus</span><br>"
            "<span style='font-size:1rem; color:#666;'>Nothing logged yet.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # --- Log input ---
    # Honour a pending clear requested by the previous run (button-submit path).
    # Must happen BEFORE the widget is instantiated — Streamlit forbids mutating
    # a widget's key after it has been rendered in the same script run.
    if st.session_state.pop("session_log_clear_pending", False):
        st.session_state["session_log_input"] = ""

    def _on_log_input_change() -> None:
        val = st.session_state.get("session_log_input", "").strip()
        if val:
            st.session_state["log_submit_pending"] = val
        # Clear the widget from inside the callback — this is the only place
        # Streamlit allows modifying the key *after* the widget is registered.
        st.session_state["session_log_input"] = ""

    input_col, btn_col = st.columns([5, 1])
    with input_col:
        new_note = st.text_input(
            "Session log",
            placeholder="What are you working on right now?",
            key="session_log_input",
            label_visibility="collapsed",
            on_change=_on_log_input_change,
        )
    with btn_col:
        log_it = st.button("Log it", key="session_log_submit", width="stretch")

    # Button click uses the current widget value; Enter key uses the captured pending value
    pending_note: str = st.session_state.pop("log_submit_pending", "")
    note_to_submit = (new_note.strip() if log_it else "") or pending_note
    if note_to_submit:
        submit_log_entry(note_to_submit)
        # Schedule input clear for next rerun regardless of submit path.
        # (For the Enter-key path the callback already cleared the widget, but
        # if the user typed ahead during the synchronous LLM extraction we want
        # to ensure the field is visibly empty once the rerun lands.)
        st.session_state["session_log_clear_pending"] = True
        st.rerun()

    # --- Collapsible log history ---
    if log_entries:
        with st.expander(f"Session log ({len(log_entries)} entries)", expanded=False):
            for entry in reversed(log_entries):
                ts = entry.timestamp.strftime("%H:%M")
                st.markdown(
                    f"<div style='padding:0.25rem 0; border-bottom:1px solid #2a2a2a; "
                    f"font-size:0.9rem;'>"
                    f"<span style='color:#888; font-size:0.8rem; margin-right:0.6rem;'>{ts}</span>"
                    f"{entry.note}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()


# ---------------------------------------------------------------------------
# Top control bar — Start / Pause / Stop
# ---------------------------------------------------------------------------


def _render_controls(open_session=None) -> None:
    """Start / Pause / Resume-session / Stop buttons in the sidebar."""
    is_monitoring = st.session_state["monitoring"]
    is_paused = st.session_state.get("paused", False)
    is_analysing = st.session_state.get("is_analysing", False)
    is_stopping = st.session_state.get("stopping", False)
    scheduler: Optional[MonitoringScheduler] = st.session_state.get("scheduler")

    errors = config.validate()
    if errors and not is_monitoring:
        for err in errors:
            st.error(err)

    if not is_monitoring:
        if open_session is not None:
            # Offer Resume (primary) and New Session (secondary)
            if st.button(
                "Resume",
                use_container_width=True,
                type="primary",
                key="ctrl_resume_session",
                disabled=bool(errors),
            ):
                resume_open_session()
                st.rerun()

            if st.button(
                "New Session",
                use_container_width=True,
                key="ctrl_new_session",
                disabled=bool(errors),
            ):
                start_monitoring()
                st.rerun()

            ts = open_session.start_time.strftime("%H:%M")
            goal_text = f" — {open_session.goal}" if open_session.goal else ""
            st.caption(f"Session from {ts}{goal_text} is ready to resume.")
        else:
            if st.button(
                "Start",
                disabled=bool(errors),
                use_container_width=True,
                type="primary",
                key="ctrl_start",
            ):
                start_monitoring()
                st.rerun()
    else:
        # Monitoring Active - Pause/Stop controls
        pause_disabled = is_analysing

        if is_paused:
            if st.button(
                "Resume",
                disabled=pause_disabled,
                use_container_width=True,
                key="ctrl_resume",
            ):
                if scheduler:
                    scheduler.resume()
                st.session_state["paused"] = False
                st.rerun()
        else:
            if st.button(
                "Pause",
                disabled=pause_disabled,
                use_container_width=True,
                key="ctrl_pause",
            ):
                if scheduler:
                    scheduler.pause()
                st.session_state["paused"] = True
                st.rerun()

        stop_label = "Stopping…" if is_stopping else "Stop"
        if st.button(
            stop_label,
            disabled=is_stopping,
            use_container_width=True,
            key="ctrl_stop",
        ):
            stop_monitoring()
            st.rerun()

    st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


@st.fragment(run_every=1)
def _sidebar_fragment() -> None:
    st.title("Coach")
    st.caption("AI focus monitoring")
    st.divider()

    st.page_link("pages/4_Settings.py", label="Settings", icon="⚙️")
    st.divider()

    is_monitoring = st.session_state["monitoring"]

    # Read open_session from session_state — it is written by _main_ui_loop
    # each render cycle so we avoid a duplicate DB call here.
    open_session = st.session_state.get("open_session")

    _render_controls(open_session)

    if is_monitoring and st.session_state.get("session_id"):
        render_session_stats()


# ---------------------------------------------------------------------------
# Main panel — unified coaching view
# ---------------------------------------------------------------------------


def _render_main_panel(open_session=None) -> None:
    is_monitoring = st.session_state["monitoring"]
    is_analysing = st.session_state["is_analysing"]
    is_paused = st.session_state.get("paused", False)
    is_stopping = st.session_state.get("stopping", False)
    result = st.session_state.get("latest_result")
    capture = st.session_state.get("latest_capture")
    scheduler: Optional[MonitoringScheduler] = st.session_state.get("scheduler")

    # --- stopping state (shown while background thread winds down) ---
    # Skip if the summary is already available — it means the session ended
    # and the summary was generated; no need to keep showing "Stopping…".
    if is_stopping and st.session_state.get("session_summary") is None:
        st.markdown(
            "<div style='text-align:center; padding:3rem 0; color:#888;'>"
            "<div style='font-size:2.4rem;'>⏹</div>"
            "<div style='font-size:1.3rem; margin-top:0.5rem; font-weight:700;'>Stopping…</div>"
            "<div style='margin-top:0.5rem;'>Finishing the current cycle and generating summary.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # --- idle state ---
    if not is_monitoring and result is None:
        if open_session is not None:
            ts = open_session.start_time.strftime("%H:%M")
            goal_text = (
                f"<div style='margin-top:0.4rem; font-size:0.95rem; color:#aaa;'>"
                f"{open_session.goal}</div>"
                if open_session.goal
                else ""
            )
            st.markdown(
                f"<div style='text-align:center; padding:4rem 0; color:#555;'>"
                f"<div style='font-size:3rem;'>▶</div>"
                f"<div style='font-size:1.4rem; margin-top:0.5rem; color:#ddd;'>"
                f"Session from {ts} ready to resume</div>"
                f"{goal_text}"
                f"<div style='margin-top:0.8rem;'>Press <strong>Resume</strong> above to continue "
                f"or <strong>New Session</strong> to start fresh.</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:center; padding:4rem 0; color:#555;'>"
                "<div style='font-size:3rem;'>⏸</div>"
                "<div style='font-size:1.4rem; margin-top:0.5rem;'>Monitoring not started</div>"
                "<div style='margin-top:0.5rem;'>Press <strong>Start</strong> above to begin.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        return

    # --- paused state ---
    if is_paused:
        st.markdown(
            "<div style='text-align:center; padding:3rem 0; color:#888;'>"
            "<div style='font-size:2.4rem;'>⏸</div>"
            "<div style='font-size:1.3rem; margin-top:0.5rem; font-weight:700;'>PAUSED</div>"
            "<div style='margin-top:0.5rem;'>Press <strong>Resume</strong> above to continue.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if result is None:
            return

    # --- analysing spinner ---
    if is_analysing:
        st.markdown(
            "<div style='text-align:center; padding:3rem 0; color:#4a90d9;'>"
            "<div style='font-size:2.4rem;'>🔍</div>"
            "<div style='font-size:1.3rem; margin-top:0.5rem;'>Analysing...</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if result is None:
        return

    score = result.focus_score
    in_rest_mode = result.mode == UserMode.REST

    accent, bg, border = score_palette(score)
    label = score_label(score)

    if in_rest_mode:
        render_rest_banner(
            st.session_state.get("latest_break_quality"),
            st.session_state.get("rest_ends_at"),
        )

    render_score_header(result, accent, label, is_monitoring, scheduler, is_paused)
    st.divider()
    render_coaching_card(result, accent, bg, border)

    if result.posture_correction:
        render_posture_callout(result.posture_correction)

    if result.suggestions:
        render_suggestion_cards(result.suggestions, border)

    if capture is not None:
        render_capture_expander(capture)


# ---------------------------------------------------------------------------
# Main panel — session summary (shown after session ends)
# ---------------------------------------------------------------------------


def _render_session_summary() -> None:
    summary = st.session_state.get("session_summary")
    if summary is None:
        return

    st.divider()
    st.subheader("Session Summary")

    score = summary.overall_score
    accent, bg, border = score_palette(score)

    st.markdown(
        f"<div style='"
        f"background:{bg}; border:1px solid {border}; border-radius:10px; "
        f"padding:1.2rem 1.8rem; text-align:center; margin-bottom:1.2rem;'>"
        f"<div style='font-size:1.3rem; font-weight:700; color:{accent}; "
        f"line-height:1.4;'>{summary.headline}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Focus", f"{summary.overall_score}/10")
    m2.metric("Deep Focus Time", f"{summary.focus_time_pct}%")
    m3.metric("Peak Period", summary.peak_period)

    st.divider()

    obs_col, act_col = st.columns(2)
    with obs_col:
        st.markdown("**Key Observations**")
        for obs in summary.key_observations:
            st.markdown(f"- {obs}")
    with act_col:
        st.markdown("**Tomorrow's Actions**")
        for action in summary.tomorrow_actions:
            st.markdown(f"- {action}")

    # Correlation insights
    if summary.correlation_insights:
        st.divider()
        st.markdown("**Correlation Insights**")
        for insight in summary.correlation_insights:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.04); border-left:3px solid #4a90d9; "
                f"padding:0.5rem 0.9rem; border-radius:4px; margin-bottom:0.4rem; "
                f"font-size:0.92rem; color:#ccc;'>{insight}</div>",
                unsafe_allow_html=True,
            )

    # Unfinished items from session log
    if summary.unfinished_items:
        st.divider()
        st.markdown("**Unfinished Items**")
        st.caption(
            "These items from your session log appear to have been left incomplete."
        )
        for item in summary.unfinished_items:
            st.markdown(
                f"<div style='background:rgba(255,180,0,0.08); border-left:3px solid #e6a817; "
                f"padding:0.5rem 0.9rem; border-radius:4px; margin-bottom:0.4rem; "
                f"font-size:0.92rem; color:#ccc;'>⚠ {item}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main panel — focus chart
# ---------------------------------------------------------------------------


def _render_focus_chart() -> None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        return

    captures = get_all_captures_for_session(session_id)
    if len(captures) < 2:
        return

    st.divider()
    st.subheader("Focus Over Time")

    data = [
        {
            "Time": rec.timestamp.strftime("%H:%M"),
            "Focus Score": rec.focus_score,
            "Mode": rec.mode_label,
        }
        for rec in captures
    ]
    df = pd.DataFrame(data)
    chart = (
        alt.Chart(df)
        .mark_line(color="#4a90d9")
        .encode(
            x=alt.X("Time:O", sort=None),
            y=alt.Y("Focus Score:Q", scale=alt.Scale(domainMin=0, domainMax=10)),
        )
        .properties(height=200)
    )
    st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------------------
# Main panel — history log
# ---------------------------------------------------------------------------


def _render_history_log() -> None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        return

    captures = get_all_captures_for_session(session_id)
    if not captures:
        return

    st.divider()
    with st.expander("Capture log", expanded=False):
        for rec in reversed(captures):
            score = rec.focus_score
            color = score_color(score)
            icon = "🔴" if rec.is_distracted else "🟢"
            ts = rec.timestamp.strftime("%H:%M:%S")
            badge_html = (
                " " + distraction_badge(rec.distraction_category)
                if rec.distraction_category
                else ""
            )
            label_html = (
                f" <span style='color:#888; font-size:0.8rem;'>[{rec.activity_label}]</span>"
                if rec.activity_label
                else ""
            )
            st.markdown(
                f"**{ts}** {icon} "
                f"<span style='color:{color}'>Focus {score}/10</span>"
                f"{label_html}{badge_html} — {rec.activity_description}",
                unsafe_allow_html=True,
            )
            st.caption(rec.feedback_message)
            if rec.posture_correction:
                st.markdown(
                    f"<span style='color:#1abc9c; font-size:0.85rem;'>"
                    f"Posture: {rec.posture_correction}</span>",
                    unsafe_allow_html=True,
                )
            if rec.suggestions:
                for s in rec.suggestions:
                    st.markdown(f"  - {s}")
            st.markdown("---")


# ---------------------------------------------------------------------------
# Polling fragment — event drain only, no rendering
#
# A minimal @st.fragment(run_every=1) that drains the scheduler event queue
# and triggers a full-page rerun whenever meaningful state changes (session
# ended, cycle complete, auth callbacks resolved).  All rendering happens in
# _main() on those full-page reruns, so the column layout and task list are
# always created at the top-level script context — satisfying Streamlit's
# fragment widget constraints and eliminating the stale-fragment-ID spam.
# ---------------------------------------------------------------------------


@st.fragment(run_every=1)
def _main_ui_loop() -> None:
    """Unified polling and rendering loop.

    Drains events every second and renders the appropriate UI (Idle or Monitoring)
    based on state.  Since this entire function is a fragment, it re-runs every
    second, ensuring the UI is always perfectly in sync with the backend state
    without needing manual st.rerun() triggers for data updates.
    """
    drain_event_queue()

    # --- Auto-resume logic ---
    # Re-attach to running session on page refresh
    was_monitoring = st.session_state.get("monitoring", False)
    auto_resume_if_needed()
    # If auto_resume changed the monitoring state, rerun immediately so the layout adapts
    if not was_monitoring and st.session_state.get("monitoring", False):
        st.rerun()
        return

    # --- Handle pending actions ---
    # Task completion triggered by a Done button in the task list.
    _pending_done = st.session_state.pop("finish_task_pending", None)
    if _pending_done is not None:
        finish_task(_pending_done)
        st.rerun()
        return

    _pending_focus = st.session_state.pop("set_focus_pending", None)
    if _pending_focus is not None:
        set_current_focus(_pending_focus)
        st.rerun()
        return

    # --- Sidebar ---
    # Handled by _sidebar_fragment in main()

    # --- Auth Servers (Spotify/Fitbit) ---
    # Check if auth completed in the background thread
    spotify_server = st.session_state.get("spotify_auth_server")
    fitbit_server = st.session_state.get("fitbit_auth_server")
    if (spotify_server is not None and spotify_server.completed) or (
        fitbit_server is not None and fitbit_server.completed
    ):
        st.rerun()
        return

    # --- Layout ---
    is_monitoring = st.session_state.get("monitoring", False)
    # Check for a resumable open session once per render (avoid duplicate DB calls).
    # Also stored in session_state so _sidebar_fragment can read it without a second DB hit.
    open_session = get_or_cleanup_open_session() if not is_monitoring else None
    st.session_state["open_session"] = open_session

    # Two-column layout: mode banner + main content on the left; current focus
    # and task list on the right.
    main_col, right_col = st.columns([7, 3])

    with main_col:
        render_mode_banner()
        _render_main_panel(open_session=open_session)
        render_errors()
        _render_session_summary()
        _render_focus_chart()
        _render_history_log()

    with right_col:
        _render_session_log()
        if is_monitoring:
            render_sprint_timer()
            render_task_list()


def main() -> None:
    try:
        _main()
    except Exception:
        logger.exception("Unhandled exception in main render loop.")
        raise


def _main() -> None:
    init_db()
    tts.ensure_voice_ready()
    init_state()

    with st.sidebar:
        _sidebar_fragment()

    # Start the unified UI loop
    _main_ui_loop()


if __name__ == "__main__":
    main()
