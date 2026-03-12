"""Streamlit session-state initialisation and preferences persistence."""

from __future__ import annotations

import streamlit as st

from coach.config import config
from coach.integrations import tts
from coach.prefs import load_prefs, save_prefs
from coach.ui.sounds import set_muted, set_volume


def persist_sound_prefs() -> None:
    """Save current sound/TTS settings from session state to disk."""
    save_prefs(
        {
            "sound_muted": st.session_state.get("sound_muted", False),
            "sound_volume": st.session_state.get("sound_volume", 0.7),
            "tts_enabled": st.session_state.get("tts_enabled", True),
        }
    )


def init_state() -> None:
    """Initialise all Streamlit session state keys exactly once."""
    # Load persisted prefs on first run
    if "prefs_loaded" not in st.session_state:
        prefs = load_prefs()
        st.session_state["sound_muted"] = prefs["sound_muted"]
        st.session_state["sound_volume"] = prefs["sound_volume"]
        st.session_state["tts_enabled"] = prefs["tts_enabled"]
        set_muted(prefs["sound_muted"])
        set_volume(prefs["sound_volume"])
        tts.set_enabled(prefs["tts_enabled"])
        st.session_state["prefs_loaded"] = True

    defaults: dict = {
        "scheduler": None,
        "session_id": None,
        "session_start": None,
        "monitoring": False,
        "stopping": False,
        "paused": False,
        "session_summary": None,
        "latest_event": None,
        "latest_capture": None,
        "latest_result": None,
        "latest_break_quality": None,
        "is_analysing": False,
        "error_message": None,
        "skipped_message": None,
        "health_check_error": None,
        "session_goal": "",
        "session_log": [],  # list[SessionLogEntry] for the current session
        "session_tasks": [],  # list[Task] extracted by LLM from session log
        "current_focus": "",  # text of the latest log entry
        "notify_mode_change": False,  # flag to fire a desktop notification
        "notify_title": "",
        "notify_body": "",
        "interval_min": config.capture_interval_min,
        "interval_max": config.capture_interval_max,
        "spotify_auth_server": None,  # AuthServer instance while OAuth is in progress
        "fitbit_auth_server": None,  # FitbitAuthServer instance while OAuth is in progress
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
