"""Desktop notification helpers for the Productivity Coach.

Two delivery paths are used together:
  1. Browser Notification API — injected via a zero-height Streamlit HTML component
     in Coach.py.
  2. Native Linux notify-send — fired from the background scheduler thread so
     notifications work even when the browser tab is in the background.

Rate-limiting: the same notification *type* is suppressed if it was sent within
the last _RATE_LIMIT_SECS seconds. This prevents spamming the user on every check-in.
"""

from __future__ import annotations

import logging
import subprocess
import time
from threading import Lock

logger = logging.getLogger(__name__)

# Minimum seconds between repeated notifications of the same type.
_RATE_LIMIT_SECS: int = 300  # ~5 min / 2 typical cycles

_last_sent: dict[str, float] = {}
_lock = Lock()


def _is_rate_limited(notification_type: str) -> bool:
    """Return True if this notification type was sent too recently."""
    with _lock:
        last = _last_sent.get(notification_type, 0.0)
        return (time.monotonic() - last) < _RATE_LIMIT_SECS


def _mark_sent(notification_type: str) -> None:
    with _lock:
        _last_sent[notification_type] = time.monotonic()


def send_native(title: str, body: str, notification_type: str) -> None:
    """
    Fire a native desktop notification via notify-send (Linux/libnotify).

    Rate-limited per *notification_type*.  Silently skips if notify-send is
    not available or if the same type was sent too recently.

    Args:
        title: Notification title (short).
        body: Notification body text.
        notification_type: Stable key used for rate-limiting, e.g. "low_focus",
            "posture", "break_quality", "mode_change".
    """
    if _is_rate_limited(notification_type):
        logger.debug(
            "Notification suppressed (rate-limited): type=%s", notification_type
        )
        return

    try:
        subprocess.run(
            ["notify-send", "--app-name=Productivity Coach", title, body],
            timeout=5,
            check=False,  # non-zero exit is not fatal
            capture_output=True,
        )
        _mark_sent(notification_type)
        logger.debug(
            "Native notification sent: type=%s title=%r", notification_type, title
        )
    except FileNotFoundError:
        logger.debug("notify-send not found — skipping native notification.")
    except Exception as exc:
        logger.warning("notify-send failed: %s", exc)
