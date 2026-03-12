"""Keyboard and mouse activity monitor using pynput.

Tracks input events between captures so that break quality can be measured
algorithmically instead of being guessed by the LLM.

Usage:
    monitor = InputMonitor()
    monitor.start()
    ...
    snapshot = monitor.snapshot_and_reset()  # call once per capture cycle
    monitor.stop()

The ``InputSnapshot`` dataclass describes the activity since the last reset:
  - keystroke_count   — number of key press events
  - mouse_distance_px — total Euclidean pixel distance the cursor travelled
  - click_count       — number of mouse button press events

Break quality score (1–10) is computed from these metrics:
  - High activity during REST → low quality score (still working)
  - Low/no activity during REST → high quality score (actually resting)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

from pynput import keyboard, mouse

logger = logging.getLogger(__name__)


@dataclass
class InputSnapshot:
    """Aggregate input activity since the last reset."""

    keystroke_count: int = 0
    mouse_distance_px: float = 0.0
    click_count: int = 0

    def compute_break_quality(self) -> int:
        """
        Return a break quality score (1–10) based on input activity.

        Scoring rationale (for a typical ~3–6 min REST window):
          - 10: completely offline — zero keystrokes, no clicks, minimal mouse movement
          - 7–9: brief glance — very few events (checking a notification, sipping water)
          - 4–6: moderate activity — scrolling, light typing
          - 1–3: heavy activity — sustained typing or clicking (still working)

        Thresholds are intentionally generous to avoid false positives on
        small incidental movements (adjusting window, picking up phone).
        """
        # Penalty points: clamp to 0–9 so score stays in 1–10 range
        penalty = 0.0

        # Typing penalty: each keystroke above a small grace threshold costs ~0.05 pts
        # 100 keystrokes ≈ a short paragraph → roughly -5 pts
        KEYSTROKE_GRACE = 5
        if self.keystroke_count > KEYSTROKE_GRACE:
            penalty += (self.keystroke_count - KEYSTROKE_GRACE) * 0.05

        # Click penalty: each click above grace → -0.5 pts
        # 10 clicks → -5 pts
        CLICK_GRACE = 2
        if self.click_count > CLICK_GRACE:
            penalty += (self.click_count - CLICK_GRACE) * 0.5

        # Mouse movement penalty: each 1000 px above grace → -1 pt
        # 5000 px total movement → -5 pts
        MOVE_GRACE_PX = 500
        if self.mouse_distance_px > MOVE_GRACE_PX:
            penalty += (self.mouse_distance_px - MOVE_GRACE_PX) / 1000.0

        score = 10 - int(min(penalty, 9))
        return max(1, min(10, score))


class InputMonitor:
    """
    Background thread that listens to keyboard and mouse events via pynput.

    Thread-safe: ``snapshot_and_reset()`` can be called from any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keystroke_count: int = 0
        self._mouse_distance_px: float = 0.0
        self._click_count: int = 0
        self._last_mouse_pos: Optional[tuple[int, int]] = None

        self._kb_listener: Optional[keyboard.Listener] = None
        self._mouse_listener: Optional[mouse.Listener] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background pynput listeners."""
        try:
            self._kb_listener = keyboard.Listener(on_press=self._on_key_press)
            self._kb_listener.daemon = True
            self._kb_listener.start()

            self._mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
            )
            self._mouse_listener.daemon = True
            self._mouse_listener.start()

            logger.info("InputMonitor started.")
        except Exception as exc:
            # pynput may fail on headless environments or without X11.
            logger.warning("InputMonitor failed to start: %s", exc)

    def stop(self) -> None:
        """Stop the background listeners."""
        if self._kb_listener:
            try:
                self._kb_listener.stop()
            except Exception:
                pass
        if self._mouse_listener:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
        logger.info("InputMonitor stopped.")

    def snapshot_and_reset(self) -> InputSnapshot:
        """
        Return the accumulated input metrics since the last call (or since start),
        then reset the counters.

        Thread-safe.
        """
        with self._lock:
            snapshot = InputSnapshot(
                keystroke_count=self._keystroke_count,
                mouse_distance_px=self._mouse_distance_px,
                click_count=self._click_count,
            )
            self._keystroke_count = 0
            self._mouse_distance_px = 0.0
            self._click_count = 0
            self._last_mouse_pos = None
        return snapshot

    # ------------------------------------------------------------------
    # pynput callbacks (called from the listener threads)
    # ------------------------------------------------------------------

    def _on_key_press(self, key: object) -> None:
        with self._lock:
            self._keystroke_count += 1

    def _on_mouse_move(self, x: int, y: int) -> None:
        with self._lock:
            if self._last_mouse_pos is not None:
                dx = x - self._last_mouse_pos[0]
                dy = y - self._last_mouse_pos[1]
                self._mouse_distance_px += math.sqrt(dx * dx + dy * dy)
            self._last_mouse_pos = (x, y)

    def _on_mouse_click(self, x: int, y: int, button: object, pressed: bool) -> None:
        if pressed:
            with self._lock:
                self._click_count += 1
