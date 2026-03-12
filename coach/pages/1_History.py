"""Session history, multi-day analytics, streaks, and weekly summary."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from coach.database import (
    get_all_captures_for_session,
    get_all_sessions_stats,
    init_db,
)
from coach.ui.utils import fmt_duration_between, hide_streamlit_chrome

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Session History — Productivity Coach",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_chrome()

# ---------------------------------------------------------------------------
# Database & data loading
# ---------------------------------------------------------------------------

init_db()

all_stats = get_all_sessions_stats()  # newest first

# ---------------------------------------------------------------------------
# Sidebar — date range filter
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Productivity Coach")
    st.caption("Session History")
    st.divider()

    if all_stats:
        dates = [s["start_time"].date() for s in all_stats]
        min_date = min(dates)
        max_date = max(dates)
    else:
        today = datetime.now(timezone.utc).date()
        min_date = max_date = today

    st.subheader("Date range")
    date_from = st.date_input(
        "From", value=min_date, min_value=min_date, max_value=max_date
    )
    date_to = st.date_input(
        "To", value=max_date, min_value=min_date, max_value=max_date
    )

    st.divider()
    st.page_link("Coach.py", label="Back to Coach", icon="🎯")

# ---------------------------------------------------------------------------
# Filter sessions by date range
# ---------------------------------------------------------------------------

filtered = [s for s in all_stats if date_from <= s["start_time"].date() <= date_to]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Session History")

if not filtered:
    st.info("No sessions recorded in the selected date range.")
    st.stop()

# ---------------------------------------------------------------------------
# Streaks & milestones (computed from all historical data)
# ---------------------------------------------------------------------------


def _compute_streaks(stats: list[dict]) -> dict:
    """Return streak info and milestone counts computed from session stats."""
    active = [s for s in reversed(stats) if s["total_captures"] > 0]
    if not active:
        return {
            "current_streak": 0,
            "longest_streak": 0,
            "total_sessions": 0,
            "total_captures": 0,
            "high_focus_sessions": 0,
            "total_focus_minutes": 0,
        }

    session_days = sorted({s["start_time"].date() for s in active})
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    current_streak = 0
    check_day = today if session_days and session_days[-1] == today else yesterday
    for day in reversed(session_days):
        if day == check_day:
            current_streak += 1
            check_day -= timedelta(days=1)
        elif day < check_day:
            break

    longest = 1
    run = 1
    for i in range(1, len(session_days)):
        if (session_days[i] - session_days[i - 1]).days == 1:
            run += 1
            longest = max(longest, run)
        else:
            run = 1

    total_captures = sum(s["total_captures"] for s in active)
    high_focus = sum(1 for s in active if s["avg_focus"] >= 7)

    total_focus_minutes = 0
    for s in active:
        if s["end_time"] and s["start_time"]:
            total_focus_minutes += int(
                (s["end_time"] - s["start_time"]).total_seconds() / 60
            )

    return {
        "current_streak": current_streak,
        "longest_streak": longest,
        "total_sessions": len(active),
        "total_captures": total_captures,
        "high_focus_sessions": high_focus,
        "total_focus_minutes": total_focus_minutes,
    }


def _score_indicator(avg: float) -> str:
    """Return a colored circle indicator based on avg focus score."""
    if avg >= 7.5:
        return "🟢"
    elif avg >= 5.0:
        return "🟡"
    else:
        return "🔴"


streaks = _compute_streaks(all_stats)

col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
col_s1.metric(
    "Current Streak",
    f"{streaks['current_streak']} day{'s' if streaks['current_streak'] != 1 else ''}",
)
col_s2.metric("Longest Streak", f"{streaks['longest_streak']} days")
col_s3.metric("Total Sessions", streaks["total_sessions"])
col_s4.metric("Total Check-ins", streaks["total_captures"])
col_s5.metric(
    "High-Focus Sessions",
    streaks["high_focus_sessions"],
    help="Sessions with avg focus >= 7/10",
)

# Milestone badges
st.markdown("")
badges = []
if streaks["current_streak"] >= 7:
    badges.append(("🔥 Week Streak", "#e67e22"))
if streaks["current_streak"] >= 30:
    badges.append(("🏆 Month Streak", "#f1c40f"))
if streaks["total_sessions"] >= 10:
    badges.append(("📅 10 Sessions", "#3498db"))
if streaks["total_sessions"] >= 50:
    badges.append(("💯 50 Sessions", "#9b59b6"))
if streaks["total_captures"] >= 100:
    badges.append(("👁 100 Check-ins", "#1abc9c"))
if streaks["high_focus_sessions"] >= 5:
    badges.append(("🎯 5 Deep Work Days", "#2ecc71"))

if badges:
    badge_html = " ".join(
        f"<span style='background:{color}; color:#fff; font-size:0.78rem; "
        f"font-weight:700; padding:3px 10px; border-radius:14px; "
        f"letter-spacing:0.04em; margin-right:4px;'>{label}</span>"
        for label, color in badges
    )
    st.markdown(badge_html, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Build daily aggregates for trend charts
# Multiple sessions on the same day are merged into one weighted daily point.
# ---------------------------------------------------------------------------

active_filtered = [s for s in filtered if s["total_captures"] > 0]

# Build a per-session row with duration in minutes
rows = []
for s in active_filtered:
    dur_min = 0
    if s["end_time"] and s["start_time"]:
        dur_min = max(0, int((s["end_time"] - s["start_time"]).total_seconds() / 60))
    rows.append(
        {
            "date": s["start_time"].date(),
            "start_dt": s["start_time"],
            "avg_focus": s["avg_focus"],
            "focused_pct": s["focused_pct"],
            "total_captures": s["total_captures"],
            "dur_min": dur_min,
            "hour_of_day": s["start_time"].hour + s["start_time"].minute / 60,
        }
    )

df_sessions = pd.DataFrame(rows) if rows else pd.DataFrame()

# Daily aggregation — weighted average focus, sum of duration & captures
if not df_sessions.empty:

    def _weighted_mean(g: pd.DataFrame, col: str) -> float:
        w = g["total_captures"]
        return float((g[col] * w).sum() / w.sum()) if w.sum() > 0 else 0.0

    daily_rows = []
    for day, g in df_sessions.groupby("date"):
        daily_rows.append(
            {
                "Date": pd.Timestamp(day),
                "Avg Focus": round(_weighted_mean(g, "avg_focus"), 2),
                "Focused %": round(_weighted_mean(g, "focused_pct"), 1),
                "Focus Minutes": int(g["dur_min"].sum()),
                "Sessions": int(len(g)),
            }
        )
    df_daily = pd.DataFrame(daily_rows).sort_values("Date").set_index("Date")
else:
    df_daily = pd.DataFrame()

# ---------------------------------------------------------------------------
# Trend charts
# ---------------------------------------------------------------------------

st.subheader("Focus Trends")

if len(df_daily) >= 2:
    tab_focus, tab_pct, tab_vol = st.tabs(
        ["Avg Focus Score", "Focused %", "Daily Focus Minutes"]
    )
    with tab_focus:
        st.caption(
            "Weighted daily average — multiple sessions on the same day are merged."
        )
        st.line_chart(
            df_daily["Avg Focus"], color="#4a90d9", y_label="Focus Score (1-10)"
        )
    with tab_pct:
        st.caption("Percentage of check-ins that were in FOCUS mode each day.")
        st.line_chart(df_daily["Focused %"], color="#2ecc71", y_label="Focused %")
    with tab_vol:
        st.caption("Total minutes spent in monitored sessions each day.")
        st.bar_chart(df_daily["Focus Minutes"], color="#9b59b6", y_label="Minutes")
elif len(df_daily) == 1:
    st.caption("Need at least 2 days of sessions to show trends.")
else:
    st.caption("No sessions with data in the selected date range.")

# ---------------------------------------------------------------------------
# "When do I work best?" scatter plot
# Shows each session as a dot: X = date, Y = time of day, color = focus score
# ---------------------------------------------------------------------------

if not df_sessions.empty and len(df_sessions) >= 3:
    st.divider()
    st.subheader("When Do You Work Best?")
    st.caption(
        "Each dot is one session. Position = when it started. "
        "Color: green = high focus (≥7), orange = medium (5-6), red = low (<5)."
    )

    _CAT_HIGH = "High Focus (≥7)"
    _CAT_MID = "Medium Focus (5-6)"
    _CAT_LOW = "Low Focus (<5)"
    _CAT_COLORS = {_CAT_HIGH: "#2ecc71", _CAT_MID: "#f39c12", _CAT_LOW: "#e74c3c"}

    scatter_rows = []
    for _, row in df_sessions.iterrows():
        score = row["avg_focus"]
        cat = _CAT_HIGH if score >= 7 else (_CAT_MID if score >= 5 else _CAT_LOW)
        scatter_rows.append(
            {
                "Date": pd.Timestamp(row["date"]),
                cat: round(row["hour_of_day"], 2),
            }
        )

    df_scatter = pd.DataFrame(scatter_rows)

    # Only pass color list for categories that actually appear in the data.
    present_cats = [
        c for c in [_CAT_HIGH, _CAT_MID, _CAT_LOW] if c in df_scatter.columns
    ]

    if present_cats:
        st.scatter_chart(
            df_scatter,
            x="Date",
            y=present_cats,
            color=[_CAT_COLORS[c] for c in present_cats],
            y_label="Hour of Day (24h)",
            height=280,
        )

st.divider()

# ---------------------------------------------------------------------------
# Session history list
# ---------------------------------------------------------------------------

st.subheader("Sessions")

for s in filtered:
    start_str = s["start_time"].strftime("%a %b %d, %Y  %H:%M")
    duration = fmt_duration_between(s["start_time"], s["end_time"])
    avg = s["avg_focus"]
    indicator = _score_indicator(avg)
    summary_text = (s.get("summary") or "").strip()
    goal_text = summary_text or (s["goal"] or "").strip()
    goal_label = "Summary" if summary_text else "Goal"

    expander_title = f"{indicator} {start_str}  —  {duration}  —  {avg}/10" + (
        f"  —  {goal_text[:60]}" if goal_text else ""
    )

    with st.expander(expander_title, expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Focus", f"{avg}/10")
        m2.metric("Focused", f"{s['focused_pct']}%")
        m3.metric("Check-ins", s["total_captures"])
        m4.metric("Duration", duration)

        if goal_text:
            st.markdown(
                f"<div style='color:#aaa; font-size:0.9rem; margin-bottom:0.6rem;'>"
                f"{goal_label}: {goal_text}</div>",
                unsafe_allow_html=True,
            )

        if s["total_captures"] > 0:
            captures = get_all_captures_for_session(s["id"])

            # Mini focus timeline — X-axis uses real datetime so points are
            # spread proportionally across the session duration.
            if len(captures) >= 2:
                chart_data = pd.DataFrame(
                    [
                        {
                            "Time": r.timestamp.replace(tzinfo=None),
                            "Focus": r.focus_score,
                        }
                        for r in captures
                    ]
                ).set_index("Time")
                st.line_chart(chart_data["Focus"], color="#4a90d9", height=150)

            # Distraction breakdown
            cats = [r.distraction_category for r in captures if r.distraction_category]
            if cats:
                cat_counts = Counter(cats)
                st.caption("Distraction breakdown:")
                cat_df = pd.DataFrame(
                    [
                        {"Category": k.replace("_", " ").title(), "Count": v}
                        for k, v in cat_counts.most_common()
                    ]
                ).set_index("Category")
                st.bar_chart(cat_df["Count"], color="#e74c3c", height=140)

        st.caption(f"Session ID: {s['id']}")

# ---------------------------------------------------------------------------
# Weekly summary
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Weekly Summary")

cutoff = datetime.now(timezone.utc) - timedelta(days=7)
weekly_sessions = [
    s for s in all_stats if s["start_time"] >= cutoff and s["total_captures"] > 0
]

if not weekly_sessions:
    st.caption("No sessions in the last 7 days to summarise.")
else:
    total_w = sum(s["total_captures"] for s in weekly_sessions)
    avg_w = round(
        sum(s["avg_focus"] * s["total_captures"] for s in weekly_sessions) / total_w, 1
    )
    st.caption(
        f"{len(weekly_sessions)} sessions · {total_w} check-ins · avg focus {avg_w}/10 "
        f"this week"
    )

    if st.button("Generate Weekly Summary", type="primary"):
        with st.spinner("Asking Gemini to analyse the week..."):
            try:
                from coach.core.agent import generate_weekly_summary

                summary = generate_weekly_summary(weekly_sessions)
                st.markdown(
                    f"<div style='background:rgba(74,144,217,0.08); border:1px solid "
                    f"rgba(74,144,217,0.3); border-radius:10px; padding:1.2rem 1.8rem; "
                    f"margin-bottom:1rem;'>"
                    f"<div style='font-size:1.2rem; font-weight:700; color:#4a90d9; "
                    f"margin-bottom:0.8rem;'>{summary.headline}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Observations**")
                    for obs in summary.observations:
                        st.markdown(f"- {obs}")
                with c2:
                    st.markdown("**Next Week's Actions**")
                    for act in summary.actions:
                        st.markdown(f"- {act}")
                if summary.patterns:
                    st.markdown("**Patterns**")
                    for p in summary.patterns:
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.04); "
                            f"border-left:3px solid #4a90d9; padding:0.5rem 0.9rem; "
                            f"border-radius:4px; margin-bottom:0.4rem; font-size:0.92rem; "
                            f"color:#ccc;'>{p}</div>",
                            unsafe_allow_html=True,
                        )
            except Exception as exc:
                st.error(f"Failed to generate weekly summary: {exc}")
