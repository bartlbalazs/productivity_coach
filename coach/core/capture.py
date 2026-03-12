"""Webcam capture utilities."""

from __future__ import annotations

import io
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import cv2
from PIL import Image

from coach.config import config
from coach.integrations.input_monitor import InputMonitor, InputSnapshot
from coach.integrations.fitbit import FitbitData, get_current_health
from coach.integrations.spotify import SpotifyTrack, get_now_playing

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Represents a single open window."""

    app_name: str  # WM_CLASS instance name, e.g. "chromium", "code"
    title: str  # Window title, e.g. "coach – agent.py – PyCharm"
    is_active: bool = False


@dataclass
class CaptureResult:
    """Holds the raw bytes for a single capture event."""

    webcam_bytes: Optional[bytes]  # JPEG bytes or None if webcam unavailable
    webcam_error: Optional[str] = None
    input_snapshot: Optional[InputSnapshot] = None  # keyboard/mouse activity
    active_window: Optional[WindowInfo] = None  # foreground window at capture time
    open_windows: list[WindowInfo] = field(default_factory=list)  # all visible windows
    spotify_track: Optional[SpotifyTrack] = None  # currently-playing track, or None
    fitbit_data: Optional[FitbitData] = None  # health snapshot, or None

    @property
    def has_webcam(self) -> bool:
        return self.webcam_bytes is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resize_image(img: Image.Image, max_dim: int) -> Image.Image:
    """Resize *img* so neither dimension exceeds *max_dim*, preserving aspect ratio."""
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _pil_to_jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _xprop(display: str, *args: str) -> str:
    """Run xprop with the given args and return stdout, or '' on any error."""
    try:
        result = subprocess.run(
            ["xprop", "-display", display, *args],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout
    except Exception:
        return ""


def _parse_wm_class(raw: str) -> str:
    """Extract the instance name from a WM_CLASS line like: WM_CLASS(STRING) = "code", "Code" """
    m = re.search(r'WM_CLASS\S*\s*=\s*"([^"]+)"', raw)
    return m.group(1) if m else ""


def _parse_window_title(raw: str) -> str:
    """Extract title from _NET_WM_NAME or WM_NAME output."""
    m = re.search(r'_NET_WM_NAME\S*\s*=\s*"([^"]+)"', raw)
    if m:
        return m.group(1)
    m = re.search(r'WM_NAME\S*\s*=\s*"([^"]+)"', raw)
    return m.group(1) if m else ""


def get_window_context() -> tuple[Optional[WindowInfo], list[WindowInfo]]:
    """
    Return (active_window, all_open_windows) using xprop via X11.

    Falls back gracefully to (None, []) when:
    - No DISPLAY is set
    - xprop is not installed
    - Running in a headless / Wayland-only environment
    """
    display = os.environ.get("DISPLAY", "")
    if not display:
        return None, []

    # Get the active window ID
    root_out = _xprop(display, "-root", "_NET_ACTIVE_WINDOW")
    active_id_match = re.search(r"window id # (0x[0-9a-f]+)", root_out)
    active_id = active_id_match.group(1) if active_id_match else ""

    # Get all stacked window IDs
    stack_out = _xprop(display, "-root", "_NET_CLIENT_LIST_STACKING")
    all_ids = re.findall(r"0x[0-9a-f]+", stack_out)

    if not all_ids:
        return None, []

    open_windows: list[WindowInfo] = []
    active_window: Optional[WindowInfo] = None

    for wid in all_ids:
        raw = _xprop(display, "-id", wid, "WM_CLASS", "_NET_WM_NAME", "WM_NAME")
        app_name = _parse_wm_class(raw)
        title = _parse_window_title(raw)
        if not app_name and not title:
            continue
        is_active = wid == active_id
        win = WindowInfo(app_name=app_name, title=title, is_active=is_active)
        open_windows.append(win)
        if is_active:
            active_window = win

    return active_window, open_windows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def capture_webcam() -> tuple[Optional[bytes], Optional[str]]:
    """
    Capture a single frame from the webcam.

    Returns:
        (jpeg_bytes, error_message) — exactly one of the two will be None.
    """
    cap = cv2.VideoCapture(config.webcam_index)
    if not cap.isOpened():
        msg = f"Could not open webcam at index {config.webcam_index}."
        logger.warning(msg)
        return None, msg

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            msg = "Failed to read frame from webcam."
            logger.warning(msg)
            return None, msg

        # OpenCV returns BGR — convert to RGB for Pillow
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img = _resize_image(img, config.webcam_max_dimension)
        jpeg_bytes = _pil_to_jpeg_bytes(img, config.image_quality)
        return jpeg_bytes, None
    except Exception as exc:
        msg = f"Webcam capture error: {exc}"
        logger.exception(msg)
        return None, msg
    finally:
        cap.release()


def capture_all(input_monitor: Optional[InputMonitor] = None) -> CaptureResult:
    """Capture a webcam frame, input activity snapshot, open window context, Spotify track, and Fitbit health data.

    If *input_monitor* is provided, its counters are snapshotted and reset as
    part of this call so that input activity is aligned with the capture timestamp.
    """
    webcam_bytes, webcam_error = capture_webcam()

    input_snapshot: Optional[InputSnapshot] = None
    if input_monitor is not None:
        input_snapshot = input_monitor.snapshot_and_reset()

    active_window, open_windows = get_window_context()

    spotify_track = get_now_playing()
    fitbit_data = get_current_health()

    return CaptureResult(
        webcam_bytes=webcam_bytes,
        webcam_error=webcam_error,
        input_snapshot=input_snapshot,
        active_window=active_window,
        open_windows=open_windows,
        spotify_track=spotify_track,
        fitbit_data=fitbit_data,
    )
