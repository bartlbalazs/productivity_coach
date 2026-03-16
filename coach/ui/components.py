"""Reusable Streamlit render functions used by Coach.py and the sidebar.

Each function is self-contained: it reads from ``st.session_state`` where
needed and calls only Streamlit rendering primitives.  No business logic
lives here — mutations are delegated to ``session_controller``.
"""

from __future__ import annotations

import csv
import html as _html
import io
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

from coach.config import config
from coach.core.agent import UserMode
from coach.integrations import tts
from coach.database import (
    get_all_captures_for_session,
    get_mode_streak_start,
    get_session_stats,
)
from coach.core.scheduler import MonitoringScheduler
from coach.core.session_state import persist_sound_prefs
from coach.ui.sounds import set_muted, set_volume
from coach.integrations.spotify import (
    AuthServer,
    disconnect as spotify_disconnect,
    get_auth_url,
    is_authenticated as spotify_is_authenticated,
    is_configured as spotify_is_configured,
    start_auth_server,
)
from coach.integrations.fitbit import (
    AuthServer as FitbitAuthServer,
    disconnect as fitbit_disconnect,
    is_authenticated as fitbit_is_authenticated,
    is_configured as fitbit_is_configured,
    start_auth_server as fitbit_start_auth_server,
)
from coach.ui.theme import (
    distraction_badge,
    score_color,
    score_label,
    score_palette,
)
from coach.ui.utils import fmt_duration_since


# ---------------------------------------------------------------------------
# Sidebar — interval settings
# ---------------------------------------------------------------------------


def render_interval_settings(is_monitoring: bool) -> None:
    """Render capture interval sliders and persist changes to session state."""
    st.subheader("Settings")
    new_min = st.slider(
        "Min interval (seconds)",
        min_value=60,
        max_value=300,
        value=st.session_state["interval_min"],
        step=30,
        disabled=is_monitoring,
        help="Minimum wait between captures.",
    )
    new_max = st.slider(
        "Max interval (seconds)",
        min_value=120,
        max_value=600,
        value=st.session_state["interval_max"],
        step=30,
        disabled=is_monitoring,
        help="Maximum wait between captures.",
    )
    if not is_monitoring:
        st.session_state["interval_min"] = new_min
        st.session_state["interval_max"] = new_max
    st.caption(f"Captures every {new_min // 60}m–{new_max // 60}m at random intervals.")


# ---------------------------------------------------------------------------
# Sidebar — sound controls
# ---------------------------------------------------------------------------


def render_sound_controls() -> None:
    """Render mute, volume, and TTS toggles; persist on change."""
    st.subheader("Sound")
    muted = st.toggle(
        "Mute sounds",
        value=st.session_state.get("sound_muted", False),
        key="sound_muted_toggle",
    )
    vol = st.slider(
        "Volume",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("sound_volume", 0.7),
        step=0.05,
        disabled=muted,
        key="sound_volume_slider",
    )
    if muted != st.session_state.get("sound_muted") or vol != st.session_state.get(
        "sound_volume"
    ):
        st.session_state["sound_muted"] = muted
        st.session_state["sound_volume"] = vol
        set_muted(muted)
        set_volume(vol)
        persist_sound_prefs()

    tts_on = st.toggle(
        "Speak instructions",
        value=st.session_state.get("tts_enabled", True),
        key="tts_enabled_toggle",
        help="Speaks coaching instructions aloud for important events (mode change, low focus, posture, poor break).",
    )
    if tts_on != st.session_state.get("tts_enabled"):
        st.session_state["tts_enabled"] = tts_on
        tts.set_enabled(tts_on)
        persist_sound_prefs()


# ---------------------------------------------------------------------------
# Sidebar — model config
# ---------------------------------------------------------------------------


def render_model_config(is_monitoring: bool) -> None:
    """Render the model name input and apply changes when not monitoring."""
    st.subheader("Model")
    model = st.text_input(
        "Model",
        value=config.model,
        disabled=is_monitoring,
        help="Vertex AI Gemini model name, e.g. gemini-3.1-pro-preview.",
    )
    if not is_monitoring and model:
        config.model = model


# ---------------------------------------------------------------------------
# Sidebar — CSV export
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sidebar — CSV export
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _build_csv_bytes(session_id: int, _cache_key: int) -> bytes:
    """Build CSV bytes for all captures in *session_id*.

    ``_cache_key`` is the current capture count; changing it busts the cache
    so the download reflects newly completed cycles without re-running on every
    fragment tick.  The leading underscore tells Streamlit not to hash this
    argument (we pass a plain int so hashing is fast regardless).
    """
    captures = get_all_captures_for_session(session_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "timestamp",
            "focus_score",
            "mode",
            "activity_label",
            "activity_description",
            "distraction_category",
            "break_quality_score",
            "posture_correction",
            "instruction",
            "suggestions",
        ]
    )
    for rec in captures:
        writer.writerow(
            [
                rec.timestamp.isoformat(),
                rec.focus_score,
                "rest" if rec.is_distracted else "focus",
                rec.activity_label or "",
                rec.activity_description,
                rec.distraction_category or "",
                rec.break_quality_score if rec.break_quality_score is not None else "",
                rec.posture_correction or "",
                rec.feedback_message,
                " | ".join(rec.suggestions),
            ]
        )
    return buf.getvalue().encode()


# ---------------------------------------------------------------------------
# Sidebar — session stats
# ---------------------------------------------------------------------------


def render_session_stats() -> None:
    """Render live session metrics and CSV export in the sidebar."""
    session_id: int = st.session_state["session_id"]
    session_start: datetime = st.session_state["session_start"]
    stats = get_session_stats(session_id)

    st.subheader("Session")
    if st.session_state.get("paused"):
        st.warning("PAUSED", icon="⏸")

    st.metric("Duration", fmt_duration_since(session_start))
    st.metric("Captures", stats["total_captures"])
    if stats["total_captures"] > 0:
        st.metric("Avg Focus", f"{stats['avg_focus']}/10")
        st.metric("Focused", f"{stats['focused_pct']}%")

    if stats["total_captures"] > 0:
        csv_bytes = _build_csv_bytes(session_id, stats["total_captures"])
        st.divider()
        st.download_button(
            label="Export CSV",
            data=csv_bytes,
            file_name=f"session_{session_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Right panel — sprint timer
# ---------------------------------------------------------------------------

_SPRINT_IDEAL_SECS = 25 * 60  # 25 minutes
_SPRINT_HARD_LIMIT_SECS = 40 * 60  # 40 minutes


def render_sprint_timer() -> None:
    """Render a circular/radial timer showing current mode streak duration.

    Displays how long the current FOCUS or REST streak has been running.
    A filled arc fills proportionally from 0 to the 40-minute hard limit,
    with a marker at 25 minutes (ideal sprint duration).  The timer updates
    every second because the outer fragment is set to ``run_every=1``.

    Only renders when monitoring is active and the session has at least one
    capture so there is a meaningful streak to display.
    """
    import math

    is_monitoring = st.session_state.get("monitoring", False)
    if not is_monitoring:
        return

    session_id: Optional[int] = st.session_state.get("session_id")
    if session_id is None:
        return

    streak_info = get_mode_streak_start(session_id)
    if streak_info is None:
        return

    mode_label, streak_start = streak_info
    now = datetime.now(timezone.utc)
    elapsed_secs = max(0, (now - streak_start).total_seconds())

    # ------------------------------------------------------------------ SVG --
    # Circular progress: radius r, centre cx/cy, full circumference = 2πr.
    # We map elapsed_secs → dashoffset so the arc grows clockwise.
    # Hard limit (40 min) = full circle.  Anything beyond is clamped.
    cx, cy, r = 80, 80, 60
    stroke_width = 10
    circumference = 2 * 3.14159265 * r
    progress = min(elapsed_secs / _SPRINT_HARD_LIMIT_SECS, 1.0)
    dash_offset = circumference * (1.0 - progress)

    # Ideal-mark angle (25/40 of full circle = 225° from top, clockwise).
    # SVG angles: 0° = 3 o'clock; we start arc at 12 o'clock (−90°).
    ideal_frac = _SPRINT_IDEAL_SECS / _SPRINT_HARD_LIMIT_SECS  # 0.625
    ideal_angle_deg = ideal_frac * 360 - 90  # degrees from 3-o'clock

    ideal_rad = math.radians(ideal_angle_deg)
    ideal_inner_x = cx + (r - stroke_width) * math.cos(ideal_rad)
    ideal_inner_y = cy + (r - stroke_width) * math.sin(ideal_rad)
    ideal_outer_x = cx + (r + stroke_width * 0.6) * math.cos(ideal_rad)
    ideal_outer_y = cy + (r + stroke_width * 0.6) * math.sin(ideal_rad)

    # Hard-limit mark is at 360° = 12 o'clock position (full circle end).
    # We place a small gap indicator at 359.9° ≈ same as 0° − a tiny tick.
    limit_rad = math.radians(360 - 90)  # = -90° = 12-o'clock = same as start
    limit_inner_x = cx + (r - stroke_width) * math.cos(limit_rad)
    limit_inner_y = cy + (r - stroke_width) * math.sin(limit_rad)
    limit_outer_x = cx + (r + stroke_width * 0.6) * math.cos(limit_rad)
    limit_outer_y = cy + (r + stroke_width * 0.6) * math.sin(limit_rad)

    # Colour: green for FOCUS, blue for REST; turns amber after ideal, red after limit.
    if mode_label == "FOCUS":
        if elapsed_secs > _SPRINT_HARD_LIMIT_SECS:
            arc_color = "#e74c3c"  # red — over hard limit
        elif elapsed_secs > _SPRINT_IDEAL_SECS:
            arc_color = "#f39c12"  # amber — past ideal but within limit
        else:
            arc_color = "#2ecc71"  # green — healthy sprint
    else:  # REST
        arc_color = "#4a90d9"  # blue

    # Elapsed time label
    elapsed_total_secs = int(elapsed_secs)
    mins_display = elapsed_total_secs // 60
    secs_display = elapsed_total_secs % 60
    time_str = f"{mins_display}m {secs_display:02d}s"

    # "since HH:MM" label
    streak_local = streak_start.astimezone()
    since_str = streak_local.strftime("%H:%M")

    mode_color = arc_color
    mode_display = mode_label

    svg = f"""
<div style="display:flex; flex-direction:column; align-items:center;
            margin-bottom:0.6rem; margin-top:0.2rem;">
  <svg width="160" height="160" viewBox="0 0 160 160"
       style="overflow:visible;">
    <!-- Track ring -->
    <circle cx="{cx}" cy="{cy}" r="{r}"
            fill="none" stroke="rgba(255,255,255,0.07)"
            stroke-width="{stroke_width}"/>
    <!-- Progress arc — starts at 12 o'clock (transform rotates −90°) -->
    <circle cx="{cx}" cy="{cy}" r="{r}"
            fill="none" stroke="{arc_color}" stroke-width="{stroke_width}"
            stroke-linecap="round"
            stroke-dasharray="{circumference:.2f}"
            stroke-dashoffset="{dash_offset:.2f}"
            transform="rotate(-90 {cx} {cy})"/>
    <!-- Ideal-mark tick (25 min) -->
    <line x1="{ideal_inner_x:.2f}" y1="{ideal_inner_y:.2f}"
          x2="{ideal_outer_x:.2f}" y2="{ideal_outer_y:.2f}"
          stroke="#f39c12" stroke-width="2.5" stroke-linecap="round"/>
    <!-- Hard-limit tick (40 min) at 12 o'clock -->
    <line x1="{limit_inner_x:.2f}" y1="{limit_inner_y:.2f}"
          x2="{limit_outer_x:.2f}" y2="{limit_outer_y:.2f}"
          stroke="#e74c3c" stroke-width="2.5" stroke-linecap="round"/>
    <!-- Centre: mode label -->
    <text x="{cx}" y="{cy - 10}" text-anchor="middle"
          font-size="11" font-weight="700" fill="{mode_color}"
          font-family="sans-serif" letter-spacing="0.08em">{mode_display}</text>
    <!-- Centre: elapsed time -->
    <text x="{cx}" y="{cy + 8}" text-anchor="middle"
          font-size="18" font-weight="800" fill="#e8e8e8"
          font-family="monospace">{time_str}</text>
    <!-- Centre: since HH:MM -->
    <text x="{cx}" y="{cy + 24}" text-anchor="middle"
          font-size="10" fill="#666"
          font-family="sans-serif">since {since_str}</text>
  </svg>
  <!-- Legend -->
  <div style="display:flex; gap:1.2rem; font-size:0.72rem; color:#666;
              margin-top:0.1rem;">
    <span><span style="color:#f39c12; font-weight:700;">|</span>&nbsp;25m ideal</span>
    <span><span style="color:#e74c3c; font-weight:700;">|</span>&nbsp;40m limit</span>
  </div>
</div>
"""
    st.markdown(svg, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Right panel — task list
# ---------------------------------------------------------------------------


def render_task_list() -> None:
    """Render the active task list in the right column of the main page.

    Reads from ``st.session_state["session_tasks"]`` (list of Task objects).
    Active tasks show a 'Done' button; completed tasks are displayed with
    a strikethrough.
    """
    tasks = st.session_state.get("session_tasks", [])

    st.markdown(
        "<div style='font-size:0.75rem; color:#888; text-transform:uppercase; "
        "letter-spacing:0.06em; margin-bottom:0.5rem;'>Tasks</div>",
        unsafe_allow_html=True,
    )

    if not tasks:
        st.markdown(
            "<div style='font-size:0.88rem; color:#555; padding:0.4rem 0;'>"
            "Log something to create a task.</div>",
            unsafe_allow_html=True,
        )
        return

    active = [t for t in tasks if not t.done]
    done_tasks = [t for t in tasks if t.done]

    for task in reversed(active):
        col_text, col_focus, col_done = st.columns([5, 1, 1])
        with col_text:
            st.markdown(
                f"<div style='padding:0.4rem 0; font-size:0.9rem; color:#e8e8e8; "
                f"border-bottom:1px solid #2a2a2a;'>{_html.escape(task.text)}</div>",
                unsafe_allow_html=True,
            )
        with col_focus:
            if st.button("▶", key=f"focus_task_{task.id}", help="Set as current focus"):
                st.session_state["set_focus_pending"] = task.text
                st.rerun()  # full-page rerun so _main() picks up the pending action
        with col_done:
            if st.button("✓", key=f"done_task_{task.id}", help="Mark as done"):
                st.session_state["finish_task_pending"] = task.id
                st.rerun()  # full-page rerun so _main() picks up the pending action

    if done_tasks:
        st.markdown(
            "<div style='margin-top:0.8rem; font-size:0.72rem; color:#444; "
            "text-transform:uppercase; letter-spacing:0.05em;'>Completed</div>",
            unsafe_allow_html=True,
        )
        for task in reversed(done_tasks):
            st.markdown(
                f"<div style='padding:0.3rem 0; font-size:0.85rem; "
                f"color:#444; text-decoration:line-through;'>{_html.escape(task.text)}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar — Spotify auth
# ---------------------------------------------------------------------------


def render_spotify_auth() -> None:
    """Render the Spotify connect/disconnect controls inside the sidebar."""
    if not spotify_is_configured():
        st.caption(
            "Set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` environment variables "
            "to enable Spotify integration."
        )
        return

    if spotify_is_authenticated():
        st.success("Connected", icon="✅")
        if st.button("Disconnect Spotify", key="spotify_disconnect"):
            spotify_disconnect()
            st.rerun()
        return

    auth_server: Optional[AuthServer] = st.session_state.get("spotify_auth_server")

    if auth_server is None:
        if st.button(
            "Connect Spotify", key="spotify_connect", use_container_width=True
        ):
            st.session_state["spotify_auth_server"] = start_auth_server()
            st.rerun()
        return

    if auth_server.completed:
        st.session_state["spotify_auth_server"] = None
        st.rerun()
        return

    if auth_server.timed_out:
        st.session_state["spotify_auth_server"] = None
        st.warning("Authorization timed out. Try again.")
        return

    if auth_server.error:
        err = auth_server.error
        st.session_state["spotify_auth_server"] = None
        st.error(f"Spotify error: {err}")
        return

    st.link_button(
        "Authorize on Spotify",
        url=get_auth_url(state=auth_server._expected_state),
        help="Opens Spotify login in a new tab. After authorizing, return here.",
    )
    st.caption("Waiting for authorization... (60 s)")


# ---------------------------------------------------------------------------
# Sidebar — Fitbit auth
# ---------------------------------------------------------------------------


def render_fitbit_auth() -> None:
    """Render the Fitbit connect/disconnect controls inside the sidebar."""
    if not fitbit_is_configured():
        st.caption(
            "Set `FITBIT_CLIENT_ID` and `FITBIT_CLIENT_SECRET` environment variables "
            "to enable Fitbit integration."
        )
        return

    if fitbit_is_authenticated():
        st.success("Connected", icon="✅")
        if st.button("Disconnect Fitbit", key="fitbit_disconnect"):
            fitbit_disconnect()
            st.rerun()
        return

    auth_server: Optional[FitbitAuthServer] = st.session_state.get("fitbit_auth_server")

    if auth_server is None:
        if st.button("Connect Fitbit", key="fitbit_connect", use_container_width=True):
            st.session_state["fitbit_auth_server"] = fitbit_start_auth_server()
            st.rerun()
        return

    if auth_server.completed:
        st.session_state["fitbit_auth_server"] = None
        st.rerun()
        return

    if auth_server.timed_out:
        st.session_state["fitbit_auth_server"] = None
        st.warning("Authorization timed out. Try again.")
        return

    if auth_server.error:
        err = auth_server.error
        st.session_state["fitbit_auth_server"] = None
        st.error(f"Fitbit error: {err}")
        return

    st.link_button(
        "Authorize on Fitbit",
        url=auth_server.auth_url,
        help="Opens Fitbit login in a new tab. After authorizing, return here.",
    )
    st.caption("Waiting for authorization... (60 s)")


# ---------------------------------------------------------------------------
# Top-of-page mode banner (FOCUS / REST)
# ---------------------------------------------------------------------------


def render_mode_banner() -> None:
    """Render a full-width coloured banner at the top of the page showing the
    current mode (FOCUS or REST).

    - Only shown when monitoring is active and at least one result exists.
    - FOCUS: vivid green gradient, score, activity label.
    - REST: vivid blue gradient, break-quality score, countdown to FOCUS switch.
    """
    result = st.session_state.get("latest_result")
    is_monitoring = st.session_state.get("monitoring", False)
    if not is_monitoring or result is None:
        return

    in_rest = result.mode == UserMode.REST  # type: ignore[attr-defined]

    if in_rest:
        # --- REST banner ---
        accent = "#4a90d9"
        bg_start = "rgba(30, 60, 114, 0.55)"
        bg_end = "rgba(42, 82, 152, 0.35)"
        border_color = "rgba(74,144,217,0.6)"
        mode_label = "REST MODE"
        score_display = ""

        bq = st.session_state.get("latest_break_quality")
        bq_html = ""
        if bq is not None:
            bq_color = "#2ecc71" if bq >= 7 else "#f39c12" if bq >= 4 else "#e74c3c"
            bq_html = (
                f"<span style='margin-left:1rem; font-size:0.9rem; color:{bq_color}; "
                f"font-weight:600;'>Break quality: {bq}/10</span>"
            )

        # Countdown to FOCUS switch
        rest_ends_at = st.session_state.get("rest_ends_at")
        time_html = ""
        if rest_ends_at is not None:
            now = datetime.now(timezone.utc)
            remaining = int((rest_ends_at - now).total_seconds())
            end_local = rest_ends_at.astimezone().strftime("%H:%M")
            if remaining > 0:
                mins, secs = divmod(remaining, 60)
                countdown = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
                time_html = (
                    f"<span style='margin-left:1.2rem; font-size:0.88rem; "
                    f"color:rgba(255,255,255,0.7);'>"
                    f"Focus resumes ~{end_local} &nbsp;·&nbsp; "
                    f"<span style='color:#fff; font-weight:700;'>{countdown}</span> remaining"
                    f"</span>"
                )
            else:
                time_html = (
                    "<span style='margin-left:1.2rem; font-size:0.88rem; "
                    "color:rgba(255,255,255,0.85); font-weight:700;'>"
                    "Switching back to FOCUS any moment…"
                    "</span>"
                )

        right_html = bq_html + time_html

    else:
        # --- FOCUS banner ---
        score = result.focus_score  # type: ignore[attr-defined]
        if score >= 8:
            accent = "#2ecc71"
            bg_start = "rgba(22, 90, 55, 0.55)"
            bg_end = "rgba(30, 110, 65, 0.35)"
            border_color = "rgba(46,204,113,0.6)"
        elif score >= 5:
            accent = "#f39c12"
            bg_start = "rgba(100, 65, 10, 0.55)"
            bg_end = "rgba(120, 80, 15, 0.35)"
            border_color = "rgba(243,156,18,0.6)"
        else:
            accent = "#e74c3c"
            bg_start = "rgba(100, 25, 20, 0.55)"
            bg_end = "rgba(120, 30, 25, 0.35)"
            border_color = "rgba(231,76,60,0.6)"

        mode_label = "FOCUS MODE"
        score_display = (
            f"<span style='font-size:1.5rem; font-weight:800; color:{accent}; "
            f"margin-left:0.8rem; vertical-align:middle;'>{score}/10</span>"
        )
        label = result.activity_label  # type: ignore[attr-defined]
        right_html = (
            f"<span style='font-size:0.88rem; color:rgba(255,255,255,0.6); "
            f"margin-left:1rem;'>{label}</span>"
            if label
            else ""
        )

    st.markdown(
        f"<div style='"
        f"background: linear-gradient(135deg, {bg_start}, {bg_end}); "
        f"border: 1px solid {border_color}; "
        f"border-radius: 10px; "
        f"padding: 0.75rem 1.4rem; "
        f"margin-bottom: 1rem; "
        f"display: flex; align-items: center;'>"
        f"<span style='font-size:1.05rem; font-weight:800; color:{accent}; "
        f"letter-spacing:0.08em; text-transform:uppercase;'>{mode_label}</span>"
        f"{score_display}"
        f"<span style='flex:1;'></span>"
        f"{right_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main panel — REST mode banner (kept for backward compat, now secondary)
# ---------------------------------------------------------------------------


def render_rest_banner(bq: Optional[int], rest_ends_at: Optional[datetime]) -> None:
    """Render the REST mode top banner with optional break-quality score and end time."""
    rest_accent = "#4a90d9"
    bq_str = f" — Break quality: {bq}/10" if bq is not None else ""

    time_str = ""
    if rest_ends_at is not None:
        now = datetime.now(timezone.utc)
        remaining_secs = int((rest_ends_at - now).total_seconds())
        end_local = rest_ends_at.astimezone().strftime("%H:%M")
        if remaining_secs > 0:
            mins, secs = divmod(remaining_secs, 60)
            countdown = (
                f"{mins}m {secs:02d}s remaining" if mins else f"{secs}s remaining"
            )
        else:
            countdown = "any moment now"
        time_str = f" &nbsp;·&nbsp; Ends ~{end_local} ({countdown})"

    st.markdown(
        f"<div style='background:rgba(74,144,217,0.10); border-left:5px solid {rest_accent}; "
        f"padding:1rem 1.4rem; border-radius:6px; margin-bottom:1.2rem;'>"
        f"<span style='font-size:1.05rem; font-weight:700; color:{rest_accent}; "
        f"letter-spacing:0.05em;'>REST MODE — Time to recover{bq_str}</span>"
        f"<span style='font-size:0.9rem; color:#aaa;'>{time_str}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main panel — score / activity header
# ---------------------------------------------------------------------------


def render_score_header(
    result: object,
    accent: str,
    label: str,
    is_monitoring: bool,
    scheduler: Optional[MonitoringScheduler],
    is_paused: bool,
) -> None:
    """Render the score/activity row and next check-in countdown."""
    col_score, col_next = st.columns([3, 1])
    with col_score:
        label_html = ""
        if result.activity_label:  # type: ignore[attr-defined]
            label_html = (
                f"<span style='background:rgba(255,255,255,0.08); color:#aaa; "
                f"font-size:0.75rem; font-weight:600; padding:2px 8px; "
                f"border-radius:10px; margin-left:0.6rem; "
                f"letter-spacing:0.03em;'>{_html.escape(result.activity_label)}</span>"  # type: ignore[attr-defined]
            )
        distraction_html = ""
        if result.distraction_category:  # type: ignore[attr-defined]
            distraction_html = (
                "<span style='margin-left:0.5rem;'>"
                + distraction_badge(result.distraction_category)  # type: ignore[attr-defined]
                + "</span>"
            )
        st.markdown(
            f"<div style='display:flex; align-items:baseline; gap:0.4rem; flex-wrap:wrap;'>"
            f"<span style='font-size:3.5rem; font-weight:800; color:{accent}; "
            f"line-height:1;'>{result.focus_score}</span>"  # type: ignore[attr-defined]
            f"<span style='font-size:1.1rem; color:#888;'>/10</span>"
            f"<span style='font-size:1.5rem; font-weight:600; color:{accent}; "
            f"margin-left:0.4rem;'>{label}</span>"
            f"{label_html}{distraction_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#aaa; font-size:0.9rem; margin-top:0.3rem;'>"
            f"{_html.escape(result.activity_description)}</div>",  # type: ignore[attr-defined]
            unsafe_allow_html=True,
        )
    with col_next:
        if is_monitoring and scheduler and not is_paused:
            secs = scheduler.seconds_until_next
            if secs is not None:
                mins, s = divmod(secs, 60)
                st.metric(
                    "Next check-in",
                    f"{mins}m {s:02d}s" if mins else f"{s}s",
                )


# ---------------------------------------------------------------------------
# Main panel — coaching card
# ---------------------------------------------------------------------------


def render_coaching_card(result: object, accent: str, bg: str, border: str) -> None:
    """Render the main coaching message card."""
    st.markdown(
        f"<div style='"
        f"background:{bg}; border:1px solid {border}; border-radius:10px; "
        f"padding:1.4rem 2rem; text-align:center; margin-bottom:1.4rem;'>"
        f"<div style='font-size:1.35rem; font-weight:700; color:{accent}; "
        f"line-height:1.4;'>{_html.escape(result.instruction)}</div>"  # type: ignore[attr-defined]
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main panel — posture callout
# ---------------------------------------------------------------------------


def render_posture_callout(correction: str) -> None:
    """Render the posture correction callout block."""
    st.markdown(
        f"<div style='background:rgba(26,188,156,0.10); border-left:4px solid #1abc9c; "
        f"border-radius:6px; padding:0.9rem 1.2rem; margin-bottom:1.2rem;'>"
        f"<div style='font-size:0.8rem; font-weight:700; color:#1abc9c; "
        f"letter-spacing:0.06em; margin-bottom:0.3rem;'>POSTURE</div>"
        f"<div style='font-size:0.95rem; color:#d0f0ea; line-height:1.5;'>"
        f"{_html.escape(correction)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main panel — suggestion cards
# ---------------------------------------------------------------------------


def render_suggestion_cards(suggestions: list[str], border: str) -> None:
    """Render suggestion cards in a column grid."""
    if not suggestions:
        return
    cols = st.columns(len(suggestions))
    for col, sug in zip(cols, suggestions):
        with col:
            st.markdown(
                f"<div style='"
                f"border:1px solid {border}; border-radius:8px; "
                f"padding:0.8rem 1rem; text-align:center; "
                f"font-size:0.95rem; font-weight:600; color:#e0e0e0; "
                f"background:rgba(255,255,255,0.03); height:100%;'>"
                f"→ {_html.escape(sug)}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main panel — latest capture expander
# ---------------------------------------------------------------------------


def render_capture_expander(capture: object) -> None:
    """Render the latest capture image inside a collapsed expander."""
    st.divider()
    with st.expander("Latest capture", expanded=False):
        if capture.has_webcam:  # type: ignore[attr-defined]
            st.image(capture.webcam_bytes, use_container_width=True)  # type: ignore[attr-defined]
        else:
            st.warning(f"Webcam unavailable: {capture.webcam_error}")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Main panel — error display
# ---------------------------------------------------------------------------


def render_errors() -> None:
    """Render health-check, analysis, and capture-skipped error messages."""
    health_err = st.session_state.get("health_check_error")
    if health_err:
        st.error(f"Cannot connect to Vertex AI — session aborted. {health_err}")

    err = st.session_state.get("error_message")
    if err:
        st.error(f"Analysis error: {err}")

    skipped = st.session_state.get("skipped_message")
    if skipped:
        st.warning(f"Capture skipped — no images available. ({skipped})")
