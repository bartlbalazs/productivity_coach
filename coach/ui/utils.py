"""Shared UI utilities for all Streamlit pages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Chrome-hiding CSS
# ---------------------------------------------------------------------------

# Streamlit chrome elements that clutter the UI: activity indicator, deploy
# button, header background.
_HIDE_CHROME_CSS = """
<style>
#MainMenu {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stStatusWidget"] {display: none;}
header[data-testid="stHeader"] {background: transparent;}
</style>
"""


def hide_streamlit_chrome() -> None:
    """Inject CSS that hides the Streamlit running-man indicator and deploy button."""
    st.markdown(_HIDE_CHROME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------


def fmt_duration(seconds: float) -> str:
    """Format a duration given in *seconds* as a human-readable string.

    Examples:
        3661  -> "1h 1m"
        125   -> "2m 5s"
        45    -> "45s"
    """
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def fmt_duration_since(start: datetime) -> str:
    """Format the elapsed time since *start* (a UTC-aware datetime)."""
    delta = datetime.now(timezone.utc) - start
    return fmt_duration(delta.total_seconds())


def fmt_duration_between(start: datetime, end: Optional[datetime]) -> str:
    """Format the duration between *start* and *end* (or now if *end* is None)."""
    finish = end or datetime.now(timezone.utc)
    delta = finish - start
    return fmt_duration(delta.total_seconds())
