"""User preferences persistence (~/.coach/prefs.json)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PREFS_PATH = os.path.join(os.path.expanduser("~"), ".coach", "prefs.json")

_DEFAULTS: dict[str, Any] = {
    "sound_muted": False,
    "sound_volume": 0.7,
    "tts_enabled": True,
}


def load_prefs() -> dict[str, Any]:
    """Load preferences from disk, filling in defaults for missing keys."""
    prefs = dict(_DEFAULTS)
    try:
        if os.path.exists(_PREFS_PATH):
            with open(_PREFS_PATH) as f:
                stored = json.load(f)
            prefs.update({k: v for k, v in stored.items() if k in _DEFAULTS})
    except Exception as exc:
        logger.warning("Could not load prefs from %s: %s", _PREFS_PATH, exc)
    return prefs


def save_prefs(prefs: dict[str, Any]) -> None:
    """Persist preferences to disk."""
    try:
        os.makedirs(os.path.dirname(_PREFS_PATH), exist_ok=True)
        with open(_PREFS_PATH, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save prefs to %s: %s", _PREFS_PATH, exc)
