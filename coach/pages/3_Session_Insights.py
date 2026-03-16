"""Session Insights — deep-dive into a single session's data and LLM summary."""

from __future__ import annotations

import html
import logging
from collections import Counter
from datetime import timedelta
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from coach.core.agent import SessionSummary, generate_summary

logger = logging.getLogger(__name__)
from coach.database import (
    CaptureRecord,
    get_all_captures_for_session,
    get_all_sessions_stats,
    get_session_log,
    get_session_stats,
    init_db,
    save_session_summary,
)
from coach.ui.utils import fmt_duration, fmt_duration_between, hide_streamlit_chrome

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Session Insights — Productivity Coach",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_chrome()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACCENT = "#4a90d9"
_GREEN = "#2ecc71"
_RED = "#e74c3c"
_LAST_CAPTURE_SPAN_MINUTES = 5


def _compute_mode_spans(captures: list[CaptureRecord]) -> pd.DataFrame:
    rows = []
    for i, rec in enumerate(captures):
        start = rec.timestamp
        end = (
            captures[i + 1].timestamp
            if i + 1 < len(captures)
            else rec.timestamp + timedelta(minutes=_LAST_CAPTURE_SPAN_MINUTES)
        )
        rows.append(
            {
                "start": start,
                "end": end,
                "mode": rec.mode_label,
                "duration_s": (end - start).total_seconds(),
            }
        )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["start", "end", "mode", "duration_s"])
    )


def _time_in_mode(spans: pd.DataFrame) -> dict[str, float]:
    if spans.empty:
        return {"FOCUS": 0.0, "REST": 0.0}
    return {
        "FOCUS": spans.loc[spans["mode"] == "FOCUS", "duration_s"].sum(),
        "REST": spans.loc[spans["mode"] == "REST", "duration_s"].sum(),
    }


def _count_mode_switches(captures: list[CaptureRecord]) -> int:
    switches = 0
    for i in range(1, len(captures)):
        if captures[i].is_distracted != captures[i - 1].is_distracted:
            switches += 1
    return switches


def _summary_from_json(raw: Optional[str]) -> Optional[SessionSummary]:
    if not raw:
        return None
    try:
        return SessionSummary.model_validate_json(raw)
    except Exception:
        logger.warning(
            "Failed to parse summary_json: %r", raw[:200] if raw else raw, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

init_db()

all_stats = get_all_sessions_stats()
sessions_with_data = [s for s in all_stats if s["total_captures"] > 0]

# ---------------------------------------------------------------------------
# Sidebar — session picker
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Productivity Coach")
    st.caption("Session Insights")
    st.divider()

    if not sessions_with_data:
        st.info("No sessions with captures recorded yet.")
        st.page_link("Coach.py", label="Back to Coach", icon="🎯")
        st.stop()

    session_options: dict[str, int] = {}
    for s in sessions_with_data:
        label = s["start_time"].strftime("%b %d  %H:%M")
        display_text = (s.get("summary") or s.get("goal") or "").strip()
        if display_text:
            label += f"  —  {display_text[:35]}"
        session_options[f"#{s['id']}  {label}"] = s["id"]

    selected_label = st.selectbox("Session", options=list(session_options.keys()))
    selected_session_id: int = session_options[selected_label]

    st.divider()
    st.page_link("Coach.py", label="Back to Coach", icon="🎯")

# ---------------------------------------------------------------------------
# Load data for selected session
# ---------------------------------------------------------------------------

captures = get_all_captures_for_session(selected_session_id)

if not captures:
    st.info("No captures recorded for this session.")
    st.stop()

session_meta = next(
    (s for s in sessions_with_data if s["id"] == selected_session_id), None
)
session_log = get_session_log(selected_session_id)
session_stats = get_session_stats(selected_session_id)

summary: Optional[SessionSummary] = _summary_from_json(
    session_meta.get("summary_json") if session_meta else None
)

spans_df = _compute_mode_spans(captures)
time_in_mode = _time_in_mode(spans_df)
mode_switches = _count_mode_switches(captures)

start_dt = captures[0].timestamp
end_dt = (
    session_meta["end_time"]
    if session_meta and session_meta.get("end_time")
    else captures[-1].timestamp
)
session_duration_s = (end_dt - start_dt).total_seconds()

break_quality_scores = [
    c.break_quality_score
    for c in captures
    if c.is_distracted and c.break_quality_score is not None
]
avg_break_quality = (
    round(sum(break_quality_scores) / len(break_quality_scores), 1)
    if break_quality_scores
    else None
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Session Insights")
st.caption(
    "Select a specific session from the sidebar to dive into detailed timelines of your "
    "focus trends, input activity (keystrokes/clicks), and distraction breakdowns. From here, "
    "you can also generate or review an AI-driven session summary to evaluate your peak "
    "periods and get actionable insights for tomorrow."
)

headline = (
    summary.headline
    if summary
    else (session_meta.get("summary") or session_meta.get("goal") or "").strip()
    if session_meta
    else ""
)
if headline:
    st.markdown(
        f"<div style='font-size:1.1rem; color:#ccc; margin-bottom:0.8rem;'>"
        f"{html.escape(headline)}</div>",
        unsafe_allow_html=True,
    )

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Duration", fmt_duration_between(start_dt, end_dt))
c2.metric("Check-ins", len(captures))
c3.metric("Avg Focus", f"{session_stats['avg_focus']}/10")
c4.metric("Focused", f"{session_stats['focused_pct']}%")
c5.metric("Mode Switches", mode_switches)
c6.metric(
    "Avg Break Quality",
    f"{avg_break_quality}/10" if avg_break_quality is not None else "—",
)

st.divider()

# ---------------------------------------------------------------------------
# Section 0 — Mode Timeline bands
# ---------------------------------------------------------------------------

st.subheader("Mode Timeline")
st.caption("Green = FOCUS · Blue = REST")

if not spans_df.empty:
    spans_plot = spans_df.copy()
    spans_plot["start_local"] = spans_plot["start"].apply(
        lambda dt: dt.astimezone(tz=None).replace(tzinfo=None)
    )
    spans_plot["end_local"] = spans_plot["end"].apply(
        lambda dt: dt.astimezone(tz=None).replace(tzinfo=None)
    )

    band_chart = (
        alt.Chart(spans_plot)
        .mark_bar(height=48, cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "start_local:T",
                title="Time",
                axis=alt.Axis(format="%H:%M", labelAngle=-30),
            ),
            x2=alt.X2("end_local:T"),
            color=alt.Color(
                "mode:N",
                scale=alt.Scale(
                    domain=["FOCUS", "REST"],
                    range=[_GREEN, _ACCENT],
                ),
                legend=alt.Legend(title="Mode"),
            ),
            tooltip=[
                alt.Tooltip("start_local:T", title="Start", format="%H:%M:%S"),
                alt.Tooltip("end_local:T", title="End", format="%H:%M:%S"),
                alt.Tooltip("mode:N", title="Mode"),
                alt.Tooltip("duration_s:Q", title="Duration (s)", format=".0f"),
            ],
        )
        .properties(height=80)
    )
    st.altair_chart(band_chart, width="stretch")
else:
    st.caption("Not enough data for timeline.")

st.divider()

# ---------------------------------------------------------------------------
# Section 1 — LLM Session Summary
# ---------------------------------------------------------------------------

if summary:
    st.subheader("Session Summary")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Scores row
        s1, s2, s3 = st.columns(3)
        s1.metric("Overall Score", f"{summary.overall_score}/10")
        s2.metric("Deep Focus", f"{summary.focus_time_pct}%")
        s3.metric("Peak Period", summary.peak_period)

        st.markdown("**Key observations**")
        for obs in summary.key_observations:
            st.markdown(f"- {obs}")

        st.markdown("**Tomorrow's actions**")
        for action in summary.tomorrow_actions:
            st.markdown(f"- {action}")

    with col_right:
        if summary.correlation_insights:
            st.markdown("**Correlation insights**")
            for insight in summary.correlation_insights:
                st.markdown(f"- {insight}")

        if summary.unfinished_items:
            st.markdown("**Unfinished items**")
            for item in summary.unfinished_items:
                st.markdown(
                    f"<div style='padding:0.3rem 0.6rem; margin-bottom:0.3rem; "
                    f"background:rgba(231,76,60,0.12); border-left:3px solid {_RED}; "
                    f"border-radius:4px; font-size:0.9rem;'>{html.escape(item)}</div>",
                    unsafe_allow_html=True,
                )
else:
    st.subheader("Session Summary")
    if session_meta and session_meta.get("end_time"):
        st.info("No AI summary generated for this session.")
        if st.button("Generate Summary"):
            with st.spinner("Analyzing session data (this may take 10-20 seconds)..."):
                try:
                    new_summary = generate_summary(selected_session_id)
                    save_session_summary(
                        selected_session_id,
                        new_summary.headline,
                        new_summary.model_dump_json(),
                    )
                    st.success("Summary generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate summary: {e}")
    else:
        st.info(
            "Session is still active or was not properly ended. Summary will be generated when stopped."
        )

    st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Focus Timeline
# ---------------------------------------------------------------------------

st.subheader("Focus Timeline")
st.caption("Green = FOCUS · Blue = REST · Dashed line = score 7")

focus_df = pd.DataFrame(
    [
        {
            "time": c.timestamp.astimezone(tz=None).replace(tzinfo=None),
            "focus_score": c.focus_score,
            "mode": c.mode_label,
            "activity": c.activity_label or c.activity_description[:40],
            "instruction": (
                c.feedback_message[:60] + "…"
                if len(c.feedback_message) > 60
                else c.feedback_message
            ),
        }
        for c in captures
    ]
)

focus_line = (
    alt.Chart(focus_df)
    .mark_line(interpolate="monotone", strokeWidth=2)
    .encode(
        x=alt.X("time:T", axis=alt.Axis(format="%H:%M", labelAngle=-30), title="Time"),
        y=alt.Y("focus_score:Q", scale=alt.Scale(domain=[0, 10]), title="Focus Score"),
        color=alt.value("#888"),
    )
)

focus_points = (
    alt.Chart(focus_df)
    .mark_circle(size=90)
    .encode(
        x=alt.X("time:T"),
        y=alt.Y("focus_score:Q", scale=alt.Scale(domain=[0, 10])),
        color=alt.Color(
            "mode:N",
            scale=alt.Scale(domain=["FOCUS", "REST"], range=[_GREEN, _ACCENT]),
            legend=alt.Legend(title="Mode"),
        ),
        tooltip=[
            alt.Tooltip("time:T", title="Time", format="%H:%M:%S"),
            alt.Tooltip("focus_score:Q", title="Focus Score"),
            alt.Tooltip("mode:N", title="Mode"),
            alt.Tooltip("activity:N", title="Activity"),
            alt.Tooltip("instruction:N", title="Instruction"),
        ],
    )
)

rule = (
    alt.Chart(pd.DataFrame({"y": [7]}))
    .mark_rule(strokeDash=[6, 4], color="#555", strokeWidth=1)
    .encode(y="y:Q")
)

st.altair_chart(
    (focus_line + focus_points + rule).properties(height=260), width="stretch"
)

# HR overlay — shown beneath the focus timeline when Fitbit data is available
hr_captures = [c for c in captures if c.heart_rate is not None]
if hr_captures:
    st.caption("Heart Rate (bpm) — from Fitbit")
    hr_df = pd.DataFrame(
        [
            {
                "time": c.timestamp.astimezone(tz=None).replace(tzinfo=None),
                "heart_rate": c.heart_rate,
                "mode": c.mode_label,
                "resting_hr": c.resting_hr,
            }
            for c in hr_captures
        ]
    )

    # Colour-code HR zones: low (<60) = blue, normal (60-100) = green, elevated (>100) = red
    hr_df["zone"] = hr_df["heart_rate"].apply(
        lambda v: "Elevated (>100)"
        if v > 100
        else ("Low (<60)" if v < 60 else "Normal (60-100)")
    )

    hr_line = (
        alt.Chart(hr_df)
        .mark_line(interpolate="monotone", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "time:T", axis=alt.Axis(format="%H:%M", labelAngle=-30), title="Time"
            ),
            y=alt.Y("heart_rate:Q", title="Heart Rate (bpm)"),
            color=alt.value("#aaa"),
        )
    )

    hr_points = (
        alt.Chart(hr_df)
        .mark_circle(size=70)
        .encode(
            x=alt.X("time:T"),
            y=alt.Y("heart_rate:Q", title="Heart Rate (bpm)"),
            color=alt.Color(
                "zone:N",
                scale=alt.Scale(
                    domain=["Low (<60)", "Normal (60-100)", "Elevated (>100)"],
                    range=["#4fc3f7", _GREEN, _RED],
                ),
                legend=alt.Legend(title="HR Zone"),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%H:%M:%S"),
                alt.Tooltip("heart_rate:Q", title="Heart Rate"),
                alt.Tooltip("zone:N", title="Zone"),
                alt.Tooltip("resting_hr:Q", title="Resting HR"),
                alt.Tooltip("mode:N", title="Mode"),
            ],
        )
    )

    # Resting HR reference line (if available)
    resting_values = [r for r in hr_df["resting_hr"] if r is not None]
    if resting_values:
        resting_val = resting_values[0]
        resting_rule = (
            alt.Chart(pd.DataFrame({"y": [resting_val]}))
            .mark_rule(strokeDash=[4, 3], color="#4fc3f7", strokeWidth=1)
            .encode(y="y:Q")
        )
        hr_chart = (hr_line + hr_points + resting_rule).properties(height=180)
    else:
        hr_chart = (hr_line + hr_points).properties(height=180)

    st.altair_chart(hr_chart, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 3b — Mode Duration Breakdown
# ---------------------------------------------------------------------------

focus_s = time_in_mode["FOCUS"]
rest_s = time_in_mode["REST"]
total_s = focus_s + rest_s

st.subheader("Mode Duration Breakdown")

if total_s > 0:
    breakdown_df = pd.DataFrame(
        [
            {
                "Mode": "FOCUS",
                "Minutes": round(focus_s / 60, 1),
                "Pct": round(focus_s / total_s * 100, 1),
            },
            {
                "Mode": "REST",
                "Minutes": round(rest_s / 60, 1),
                "Pct": round(rest_s / total_s * 100, 1),
            },
        ]
    )

    breakdown_bar = (
        alt.Chart(breakdown_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Minutes:Q", title="Minutes"),
            y=alt.Y("Mode:N", title=None, sort=["FOCUS", "REST"]),
            color=alt.Color(
                "Mode:N",
                scale=alt.Scale(
                    domain=["FOCUS", "REST"],
                    range=[_GREEN, _ACCENT],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Mode:N"),
                alt.Tooltip("Minutes:Q", title="Minutes", format=".1f"),
                alt.Tooltip("Pct:Q", title="%", format=".1f"),
            ],
        )
        .properties(height=100)
    )
    st.altair_chart(breakdown_bar, width="stretch")

    col_f, col_r = st.columns(2)
    col_f.metric(
        "Time in FOCUS",
        fmt_duration(focus_s),
        delta=f"{breakdown_df.loc[breakdown_df['Mode'] == 'FOCUS', 'Pct'].iloc[0]:.0f}%",
    )
    col_r.metric(
        "Time in REST",
        fmt_duration(rest_s),
        delta=f"{breakdown_df.loc[breakdown_df['Mode'] == 'REST', 'Pct'].iloc[0]:.0f}%",
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Break Quality
# ---------------------------------------------------------------------------

bq_captures = [
    c for c in captures if c.is_distracted and c.break_quality_score is not None
]
if bq_captures:
    st.subheader("Break Quality")
    st.caption("1 = poor rest · 10 = great rest · green ≥ 7")

    bq_df = pd.DataFrame(
        [
            {
                "time": c.timestamp.astimezone(tz=None).replace(tzinfo=None),
                "break_quality": c.break_quality_score,
                "activity": c.activity_label or c.activity_description[:40],
            }
            for c in bq_captures
        ]
    )

    bq_bars = (
        alt.Chart(bq_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "time:T", axis=alt.Axis(format="%H:%M", labelAngle=-30), title="Time"
            ),
            y=alt.Y(
                "break_quality:Q",
                scale=alt.Scale(domain=[0, 10]),
                title="Break Quality",
            ),
            color=alt.condition(
                alt.datum.break_quality >= 7,
                alt.value(_GREEN),
                alt.value(_RED),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%H:%M:%S"),
                alt.Tooltip("break_quality:Q", title="Break Quality"),
                alt.Tooltip("activity:N", title="Activity"),
            ],
        )
        .properties(height=160)
    )
    st.altair_chart(bq_bars, width="stretch")
    st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Distraction Breakdown
# ---------------------------------------------------------------------------

distraction_cats = [c.distraction_category for c in captures if c.distraction_category]
if distraction_cats:
    st.subheader("Distraction Breakdown")
    total_distracted = sum(1 for c in captures if c.is_distracted)
    st.caption(
        f"{total_distracted} distracted check-ins out of {len(captures)} total "
        f"({round(total_distracted / len(captures) * 100)}%)"
    )

    cat_counts = Counter(distraction_cats)
    cat_df = pd.DataFrame(
        [
            {"Category": k.replace("_", " ").title(), "Count": v}
            for k, v in cat_counts.most_common()
        ]
    )

    dist_bar = (
        alt.Chart(cat_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Category:N", sort="-y", title=None),
            y=alt.Y("Count:Q", title="Check-ins"),
            color=alt.value(_RED),
            tooltip=[
                alt.Tooltip("Category:N"),
                alt.Tooltip("Count:Q"),
            ],
        )
        .properties(height=180)
    )
    st.altair_chart(dist_bar, width="stretch")
    st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Input Activity
# ---------------------------------------------------------------------------

input_captures = [
    c
    for c in captures
    if c.keystroke_count is not None
    or c.click_count is not None
    or c.mouse_distance_px is not None
]
if input_captures:
    st.subheader("Input Activity")
    st.caption("Keyboard, mouse clicks, and mouse movement per check-in")

    input_df = pd.DataFrame(
        [
            {
                "time": c.timestamp.astimezone(tz=None).replace(tzinfo=None),
                "Keystrokes": c.keystroke_count or 0,
                "Clicks": c.click_count or 0,
                "Mouse Distance (px)": int(c.mouse_distance_px or 0),
            }
            for c in input_captures
        ]
    )

    ia1, ia2, ia3 = st.columns(3)

    with ia1:
        st.caption("Keystrokes")
        ks_df = input_df[["time", "Keystrokes"]].rename(columns={"Keystrokes": "value"})
        st.altair_chart(
            alt.Chart(ks_df)
            .mark_area(interpolate="monotone", color=_ACCENT, opacity=0.6)
            .encode(
                x=alt.X("time:T", axis=alt.Axis(format="%H:%M"), title=None),
                y=alt.Y("value:Q", title=None),
                tooltip=[
                    alt.Tooltip("time:T", format="%H:%M:%S"),
                    alt.Tooltip("value:Q", title="Keystrokes"),
                ],
            )
            .properties(height=120),
            width="stretch",
        )

    with ia2:
        st.caption("Clicks")
        cl_df = input_df[["time", "Clicks"]].rename(columns={"Clicks": "value"})
        st.altair_chart(
            alt.Chart(cl_df)
            .mark_area(interpolate="monotone", color=_GREEN, opacity=0.6)
            .encode(
                x=alt.X("time:T", axis=alt.Axis(format="%H:%M"), title=None),
                y=alt.Y("value:Q", title=None),
                tooltip=[
                    alt.Tooltip("time:T", format="%H:%M:%S"),
                    alt.Tooltip("value:Q", title="Clicks"),
                ],
            )
            .properties(height=120),
            width="stretch",
        )

    with ia3:
        st.caption("Mouse Distance (px)")
        md_df = input_df[["time", "Mouse Distance (px)"]].rename(
            columns={"Mouse Distance (px)": "value"}
        )
        st.altair_chart(
            alt.Chart(md_df)
            .mark_area(interpolate="monotone", color="#f39c12", opacity=0.6)
            .encode(
                x=alt.X("time:T", axis=alt.Axis(format="%H:%M"), title=None),
                y=alt.Y("value:Q", title=None),
                tooltip=[
                    alt.Tooltip("time:T", format="%H:%M:%S"),
                    alt.Tooltip("value:Q", title="Mouse Distance (px)"),
                ],
            )
            .properties(height=120),
            width="stretch",
        )

    st.divider()

# ---------------------------------------------------------------------------
# Section 6 — Session Log
# ---------------------------------------------------------------------------

if session_log:
    st.subheader("Session Log")
    for entry in session_log:
        ts = entry.timestamp.astimezone(tz=None).strftime("%H:%M")
        st.markdown(
            f"<div style='padding:0.3rem 0; border-bottom:1px solid #2a2a2a; font-size:0.9rem;'>"
            f"<span style='color:#888; font-size:0.8rem; margin-right:0.8rem;'>{ts}</span>"
            f"{html.escape(entry.note)}</div>",
            unsafe_allow_html=True,
        )
    st.divider()

# ---------------------------------------------------------------------------
# Section 6b — Capture Log
# ---------------------------------------------------------------------------

st.subheader("Capture Log")

log_rows = []
for c in captures:
    log_rows.append(
        {
            "Time": c.timestamp.astimezone(tz=None).strftime("%H:%M:%S"),
            "Mode": c.mode_label,
            "Score": c.focus_score,
            "Activity": c.activity_label or "—",
            "Instruction": c.feedback_message,
            "Break Quality": (
                c.break_quality_score
                if c.is_distracted and c.break_quality_score is not None
                else None
            ),
            "Posture": "⚠" if c.posture_correction else "✓",
        }
    )

log_df = pd.DataFrame(log_rows)

st.dataframe(
    log_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Score": st.column_config.NumberColumn(min_value=1, max_value=10),
        "Mode": st.column_config.TextColumn(),
    },
)

st.divider()

# ---------------------------------------------------------------------------
# Section 7 — Cross-Session Trends
# ---------------------------------------------------------------------------

st.subheader("Cross-Session Trends")

trend_sessions = [s for s in all_stats if s["total_captures"] > 0][-30:]

if len(trend_sessions) < 2:
    st.caption("Not enough sessions to show trends yet.")
else:
    trend_df = pd.DataFrame(
        [
            {
                "session": f"#{s['id']} {s['start_time'].strftime('%b %d')}",
                "date": s["start_time"].astimezone(tz=None).replace(tzinfo=None),
                "avg_focus": s["avg_focus"],
                "focused_pct": s["focused_pct"],
                "is_selected": s["id"] == selected_session_id,
            }
            for s in trend_sessions
        ]
    )

    overall_avg_focus = round(
        sum(s["avg_focus"] for s in trend_sessions) / len(trend_sessions), 1
    )
    overall_avg_focused_pct = round(
        sum(s["focused_pct"] for s in trend_sessions) / len(trend_sessions), 1
    )

    this_session = next(
        (s for s in trend_sessions if s["id"] == selected_session_id), None
    )

    if this_session:
        t1, t2 = st.columns(2)
        t1.metric(
            "Avg Focus vs. your average",
            f"{this_session['avg_focus']}/10",
            delta=f"{round(this_session['avg_focus'] - overall_avg_focus, 1):+.1f} vs avg {overall_avg_focus}",
        )
        t2.metric(
            "Focused % vs. your average",
            f"{this_session['focused_pct']}%",
            delta=f"{round(this_session['focused_pct'] - overall_avg_focused_pct, 1):+.1f}% vs avg {overall_avg_focused_pct}%",
        )

    # Focus score trend line
    base = alt.Chart(trend_df)

    focus_trend_line = base.mark_line(
        interpolate="monotone", strokeWidth=2, color=_ACCENT
    ).encode(
        x=alt.X("date:T", axis=alt.Axis(format="%b %d", labelAngle=-30), title=None),
        y=alt.Y("avg_focus:Q", scale=alt.Scale(domain=[0, 10]), title="Avg Focus"),
        tooltip=[
            alt.Tooltip("session:N", title="Session"),
            alt.Tooltip("avg_focus:Q", title="Avg Focus"),
        ],
    )

    focus_dots = base.mark_circle(size=70).encode(
        x=alt.X("date:T"),
        y=alt.Y("avg_focus:Q", scale=alt.Scale(domain=[0, 10])),
        color=alt.condition(
            alt.datum.is_selected,
            alt.value("#f39c12"),
            alt.value(_ACCENT),
        ),
        size=alt.condition(alt.datum.is_selected, alt.value(150), alt.value(60)),
        tooltip=[
            alt.Tooltip("session:N", title="Session"),
            alt.Tooltip("avg_focus:Q", title="Avg Focus"),
        ],
    )

    avg_rule = (
        alt.Chart(pd.DataFrame({"y": [overall_avg_focus]}))
        .mark_rule(strokeDash=[4, 3], color="#555", strokeWidth=1)
        .encode(y="y:Q")
    )

    st.caption(
        "Avg focus per session — orange dot = this session · dashed = your average"
    )
    st.altair_chart(
        (focus_trend_line + focus_dots + avg_rule).properties(height=200),
        width="stretch",
    )

    # Focused % trend
    pct_line = base.mark_line(
        interpolate="monotone", strokeWidth=2, color=_GREEN
    ).encode(
        x=alt.X("date:T", axis=alt.Axis(format="%b %d", labelAngle=-30), title=None),
        y=alt.Y("focused_pct:Q", scale=alt.Scale(domain=[0, 100]), title="Focused %"),
        tooltip=[
            alt.Tooltip("session:N", title="Session"),
            alt.Tooltip("focused_pct:Q", title="Focused %"),
        ],
    )

    pct_dots = base.mark_circle(size=70).encode(
        x=alt.X("date:T"),
        y=alt.Y("focused_pct:Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.condition(
            alt.datum.is_selected,
            alt.value("#f39c12"),
            alt.value(_GREEN),
        ),
        size=alt.condition(alt.datum.is_selected, alt.value(150), alt.value(60)),
        tooltip=[
            alt.Tooltip("session:N", title="Session"),
            alt.Tooltip("focused_pct:Q", title="Focused %"),
        ],
    )

    pct_rule = (
        alt.Chart(pd.DataFrame({"y": [overall_avg_focused_pct]}))
        .mark_rule(strokeDash=[4, 3], color="#555", strokeWidth=1)
        .encode(y="y:Q")
    )

    st.caption(
        "Focused % per session — orange dot = this session · dashed = your average"
    )
    st.altair_chart(
        (pct_line + pct_dots + pct_rule).properties(height=200),
        width="stretch",
    )
