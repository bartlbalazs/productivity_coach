"""Pure UI theme helpers: colours, score labels, and HTML badge fragments.

No Streamlit imports — safe to import from any context.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Score thresholds
# ---------------------------------------------------------------------------

SCORE_HIGH = 8
SCORE_MID = 5
SCORE_MOSTLY = 6
SCORE_PARTIAL = 4

# ---------------------------------------------------------------------------
# Distraction category colours
# ---------------------------------------------------------------------------

DISTRACTION_COLORS: dict[str, str] = {
    "social_media": "#e74c3c",
    "messaging": "#e67e22",
    "browsing": "#f39c12",
    "video": "#9b59b6",
    "phone": "#3498db",
    "conversation": "#1abc9c",
    "fatigue": "#95a5a6",
    "environment": "#7f8c8d",
    "unknown": "#555",
}

# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def score_color(score: int) -> str:
    """Return a single hex colour string for the given focus score."""
    if score >= SCORE_HIGH:
        return "#2ecc71"
    if score >= SCORE_MID:
        return "#f39c12"
    return "#e74c3c"


def score_palette(score: int) -> tuple[str, str, str]:
    """Return (accent, bg, border) colour strings for a focus score."""
    if score >= SCORE_HIGH:
        return "#2ecc71", "rgba(46,204,113,0.08)", "rgba(46,204,113,0.4)"
    if score >= SCORE_MID:
        return "#f39c12", "rgba(243,156,18,0.08)", "rgba(243,156,18,0.4)"
    return "#e74c3c", "rgba(231,76,60,0.10)", "rgba(231,76,60,0.5)"


def score_label(score: int) -> str:
    """Return a human-readable focus label for the given score."""
    if score >= SCORE_HIGH:
        return "Deeply Focused"
    if score >= SCORE_MOSTLY:
        return "Mostly Focused"
    if score >= SCORE_PARTIAL:
        return "Partially Focused"
    return "Distracted"


# ---------------------------------------------------------------------------
# HTML fragment helpers
# ---------------------------------------------------------------------------


def distraction_badge(category: str) -> str:
    """Return an inline HTML badge element for a distraction category."""
    color = DISTRACTION_COLORS.get(category, "#555")
    label = category.replace("_", " ").title()
    return (
        f"<span style='background:{color}; color:#fff; font-size:0.75rem; "
        f"font-weight:700; padding:2px 8px; border-radius:12px; "
        f"letter-spacing:0.04em;'>{label}</span>"
    )
