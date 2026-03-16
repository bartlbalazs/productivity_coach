"""Settings page for the Productivity Coach application."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path so we can import 'coach' package
# This is needed if running this page directly via Streamlit multipage
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from coach.ui.components import (
    render_fitbit_auth,
    render_interval_settings,
    render_model_config,
    render_sound_controls,
    render_spotify_auth,
)
from coach.ui.utils import hide_streamlit_chrome
from coach.core.session_state import init_state

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Settings - Productivity Coach",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_chrome()

# Initialise session state so interval sliders have their keys populated
init_state()

# ---------------------------------------------------------------------------
# Page content — executed directly by Streamlit's multipage runner
# ---------------------------------------------------------------------------

st.title("Settings")

# Navigation back to home
st.page_link("Coach.py", label="Back to Dashboard", icon="🏠")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("General")
    # Interval sliders are disabled while monitoring is active to avoid
    # changing the scheduler interval mid-session.
    is_monitoring = st.session_state.get("monitoring", False)

    render_interval_settings(is_monitoring)
    st.divider()
    render_sound_controls()
    st.divider()
    render_model_config(is_monitoring)

with col2:
    st.subheader("Integrations")
    st.write("### Spotify")
    render_spotify_auth()
    st.divider()
    st.write("### Fitbit")
    render_fitbit_auth()
