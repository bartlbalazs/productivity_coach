"""Achievements page — gamified badge system for the Productivity Coach."""

from __future__ import annotations

import html as _html
import streamlit as st

from coach.achievements import (
    ACHIEVEMENTS,
    CATEGORY_META,
    Achievement,
    evaluate_achievements,
    get_progress,
)
from coach.database import get_achievement_stats, init_db
from coach.ui.utils import hide_streamlit_chrome

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Achievements — Productivity Coach",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_chrome()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def _load_stats() -> dict:
    return get_achievement_stats()


init_db()
try:
    stats = _load_stats()
    unlocked = evaluate_achievements(stats)
except Exception as exc:
    st.error(f"Failed to load achievement data: {exc}")
    st.stop()

total = len(ACHIEVEMENTS)
earned = len(unlocked)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Productivity Coach")
    st.caption("Achievements")
    st.divider()

    # overall progress ring / bar
    pct = int(earned / total * 100) if total else 0
    st.markdown(
        f"""
<div style="text-align:center; padding:0.6rem 0 1rem;">
  <div style="font-size:3rem; font-weight:800; color:#f1c40f; line-height:1;">{earned}</div>
  <div style="font-size:0.85rem; color:#888;">of {total} unlocked</div>
  <div style="margin-top:0.6rem; background:#1e1e1e; border-radius:8px;
              height:8px; overflow:hidden;">
    <div style="height:100%; width:{pct}%;
                background:linear-gradient(90deg,#f1c40f,#e67e22);
                border-radius:8px; transition:width 0.4s;"></div>
  </div>
  <div style="font-size:0.75rem; color:#666; margin-top:0.3rem;">{pct}% complete</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    # Per-category quick stats
    st.markdown(
        "<div style='font-size:0.75rem; color:#888; text-transform:uppercase; "
        "letter-spacing:0.05em; margin-bottom:0.4rem;'>By Category</div>",
        unsafe_allow_html=True,
    )
    for cat_id, (cat_label, cat_icon, cat_color) in CATEGORY_META.items():
        cat_total = sum(1 for a in ACHIEVEMENTS if a.category == cat_id)
        cat_earned = sum(
            1 for a in ACHIEVEMENTS if a.category == cat_id and a.id in unlocked
        )
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; "
            f"font-size:0.88rem; padding:0.15rem 0;'>"
            f"<span>{cat_icon} {cat_label}</span>"
            f"<span style='color:{cat_color}; font-weight:700;'>{cat_earned}/{cat_total}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.page_link("Coach.py", label="Back to Coach", icon="🎯")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🏅 Achievements")
st.caption(
    "Browse the gamified categories to track your unlocked badges and monitor your progress "
    "bars for long-term milestones. Check the bottom of the page to discover your next closest "
    "unlockable achievement to help maintain your momentum."
)

# Motivational headline bar
if earned == 0:
    headline = (
        "🚀 Your journey starts now — complete your first session to earn a badge."
    )
    headline_color = "#4a90d9"
elif pct < 25:
    headline = f"🌱 Great start! You've unlocked {earned} {'badge' if earned == 1 else 'badges'} — the momentum is building."
    headline_color = "#2ecc71"
elif pct < 50:
    headline = f"🔥 Solid progress — {earned} badges earned. You're in the top half!"
    headline_color = "#f39c12"
elif pct < 75:
    headline = f"⚡ Impressive! {earned} badges. You're outworking most people."
    headline_color = "#e67e22"
elif pct < 100:
    headline = f"🏆 Elite level — {earned}/{total} badges. Almost complete!"
    headline_color = "#f1c40f"
else:
    headline = (
        "👑 LEGENDARY — All achievements unlocked. You are the Productivity Coach."
    )
    headline_color = "#f1c40f"

st.markdown(
    f"<div style='background:rgba(255,255,255,0.04); border-left:4px solid {headline_color}; "
    f"border-radius:6px; padding:0.85rem 1.3rem; margin-bottom:1.6rem; "
    f"font-size:1.02rem; font-weight:600; color:#e8e8e8;'>{headline}</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOCKED_ICON = "🔒"


def _render_card(ach: Achievement, is_unlocked: bool, stats: dict) -> None:
    """Render a single achievement card."""
    is_secret_locked = ach.secret and not is_unlocked

    if is_secret_locked:
        icon = _LOCKED_ICON
        title = "???"
        desc = "Keep working to reveal this secret achievement."
        card_opacity = "0.4"
        border_color = "rgba(255,255,255,0.06)"
        icon_bg = "rgba(255,255,255,0.05)"
        title_color = "#555"
    elif is_unlocked:
        icon = ach.icon
        title = _html.escape(ach.title)
        desc = _html.escape(ach.description)
        card_opacity = "1"
        border_color = CATEGORY_META[ach.category][2]
        icon_bg = f"rgba(255,255,255,0.07)"
        title_color = "#f0f0f0"
    else:
        icon = ach.icon
        title = _html.escape(ach.title)
        desc = _html.escape(ach.description)
        card_opacity = "0.45"
        border_color = "rgba(255,255,255,0.08)"
        icon_bg = "rgba(255,255,255,0.03)"
        title_color = "#666"

    # Unlocked glow
    glow = (
        f"box-shadow:0 0 14px {CATEGORY_META[ach.category][2]}55;"
        if is_unlocked
        else ""
    )

    # Progress bar (for cumulative achievements)
    progress_html = ""
    if not is_secret_locked and not is_unlocked:
        prog = get_progress(ach, stats)
        if prog is not None:
            cur, tgt = prog
            bar_pct = min(int(cur / tgt * 100), 100) if tgt else 0
            cat_color = CATEGORY_META[ach.category][2]
            progress_html = (
                f"<div style='margin-top:0.5rem;'>"
                f"<div style='font-size:0.7rem; color:#666; margin-bottom:2px;'>"
                f"{cur:,} / {tgt:,}</div>"
                f"<div style='background:#1a1a1a; border-radius:4px; height:5px; overflow:hidden;'>"
                f"<div style='width:{bar_pct}%; height:100%; background:{cat_color}; "
                f"border-radius:4px;'></div></div></div>"
            )

    unlocked_badge = ""
    if is_unlocked and not ach.secret:
        cat_color = CATEGORY_META[ach.category][2]
        unlocked_badge = (
            f"<div style='font-size:0.65rem; font-weight:700; color:{cat_color}; "
            f"text-transform:uppercase; letter-spacing:0.06em; margin-top:0.3rem;'>"
            f"✓ Unlocked</div>"
        )
    elif is_unlocked and ach.secret:
        unlocked_badge = (
            "<div style='font-size:0.65rem; font-weight:700; color:#9b59b6; "
            "text-transform:uppercase; letter-spacing:0.06em; margin-top:0.3rem;'>"
            "✓ Secret Unlocked!</div>"
        )

    card_style = (
        f"opacity:{card_opacity}; border:1px solid {border_color}; border-radius:12px;"
        f" padding:1rem 1.1rem 0.9rem; background:rgba(255,255,255,0.02);"
        f" height:100%; {glow}"
    )
    icon_div_style = (
        f"font-size:1.6rem; line-height:1; background:{icon_bg};"
        f" border-radius:8px; padding:0.3rem 0.4rem; min-width:2.2rem; text-align:center;"
    )
    st.markdown(
        f'<div style="{card_style}">'
        f'<div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.5rem;">'
        f'<div style="{icon_div_style}">{icon}</div>'
        f'<div style="font-size:0.9rem; font-weight:700; color:{title_color}; line-height:1.25;">{title}</div>'
        f"</div>"
        f'<div style="font-size:0.78rem; color:#666; line-height:1.45;">{desc}</div>'
        f"{progress_html}"
        f"{unlocked_badge}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Render categories
# ---------------------------------------------------------------------------

COLS_PER_ROW = 4

for cat_id, (cat_label, cat_icon, cat_color) in CATEGORY_META.items():
    cat_achievements = [a for a in ACHIEVEMENTS if a.category == cat_id]
    cat_earned = sum(1 for a in cat_achievements if a.id in unlocked)
    cat_total = len(cat_achievements)

    st.markdown(
        f"<div style='display:flex; align-items:baseline; gap:0.6rem; "
        f"margin-top:1.8rem; margin-bottom:0.8rem;'>"
        f"<span style='font-size:1.35rem;'>{cat_icon}</span>"
        f"<span style='font-size:1.2rem; font-weight:700; color:#e8e8e8;'>{cat_label}</span>"
        f"<span style='font-size:0.78rem; color:{cat_color}; font-weight:600; "
        f"background:rgba(255,255,255,0.06); padding:1px 8px; border-radius:10px;'>"
        f"{cat_earned}/{cat_total}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Sort: unlocked first, then locked non-secret, then locked secret
    def _sort_key(a: Achievement) -> tuple:
        if a.id in unlocked:
            return (0, a.title)
        if not a.secret:
            return (1, a.title)
        return (2, a.title)

    sorted_ach = sorted(cat_achievements, key=_sort_key)

    # Render in grid
    for row_start in range(0, len(sorted_ach), COLS_PER_ROW):
        row = sorted_ach[row_start : row_start + COLS_PER_ROW]
        cols = st.columns(COLS_PER_ROW)
        for col, ach in zip(cols, row):
            with col:
                _render_card(ach, ach.id in unlocked, stats)
        # empty spacer between rows
        st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    st.divider()

# ---------------------------------------------------------------------------
# Footer — motivational callout
# ---------------------------------------------------------------------------

remaining = total - earned
if remaining > 0:
    # Show the locked non-secret achievement closest to completion.
    # Progress-tracked ones are scored by current/target; the rest score 0.
    locked = [a for a in ACHIEVEMENTS if a.id not in unlocked and not a.secret]
    if locked:

        def _completion_score(a: Achievement) -> float:
            prog = get_progress(a, stats)
            if prog is None:
                return 0.0
            current, target = prog
            return current / target if target else 0.0

        next_ach = max(locked, key=_completion_score)
        cat_color = CATEGORY_META[next_ach.category][2]
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.03); border:1px solid "
            f"rgba(255,255,255,0.08); border-radius:10px; padding:1rem 1.5rem; "
            f"display:flex; align-items:center; gap:1rem;'>"
            f"<span style='font-size:2rem;'>{next_ach.icon}</span>"
            f"<div>"
            f"<div style='font-size:0.75rem; color:#888; text-transform:uppercase; "
            f"letter-spacing:0.06em;'>Next to unlock</div>"
            f"<div style='font-size:1rem; font-weight:700; color:{cat_color};'>"
            f"{_html.escape(next_ach.title)}</div>"
            f"<div style='font-size:0.82rem; color:#777;'>{_html.escape(next_ach.description)}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
