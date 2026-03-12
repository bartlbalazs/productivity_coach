"""Programmatically generated audio cues for key coaching events.

All sounds are synthesised on first use (no audio files required) and played
in a fire-and-forget daemon thread so callers never block.

Volume (0.0–1.0) and mute state are controlled via set_volume() / set_muted().
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 44100
_pygame_lock = threading.Lock()
_pygame_ready = False

# Runtime-adjustable playback settings (set by the UI)
_volume: float = 0.7
_muted: bool = False
_settings_lock = threading.Lock()


def set_volume(volume: float) -> None:
    """Set master volume (0.0–1.0)."""
    global _volume
    with _settings_lock:
        _volume = max(0.0, min(1.0, volume))


def set_muted(muted: bool) -> None:
    """Mute or unmute all sounds."""
    global _muted
    with _settings_lock:
        _muted = muted


def get_volume() -> float:
    with _settings_lock:
        return _volume


def get_muted() -> bool:
    with _settings_lock:
        return _muted


def _init_pygame() -> bool:
    """Initialise pygame mixer exactly once. Returns True if usable."""
    global _pygame_ready
    if _pygame_ready:
        return True
    with _pygame_lock:
        if _pygame_ready:
            return True
        try:
            import pygame

            pygame.mixer.init(frequency=_SAMPLE_RATE, size=-16, channels=1, buffer=512)
            _pygame_ready = True
            logger.debug("pygame mixer initialised at %d Hz.", _SAMPLE_RATE)
            return True
        except Exception as exc:
            logger.warning("Audio unavailable (pygame init failed): %s", exc)
            return False


# ---------------------------------------------------------------------------
# Waveform helpers
# ---------------------------------------------------------------------------


def _sine(freq: float, duration: float, volume: float = 0.4) -> np.ndarray:
    """Return a mono sine-wave array (int16)."""
    t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    # Apply a short linear fade-out to avoid clicks
    fade_samples = min(int(_SAMPLE_RATE * 0.02), len(wave))
    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return (wave * volume * 32767).astype(np.int16)


def _concat(*arrays: np.ndarray, gap_ms: int = 0) -> np.ndarray:
    """Concatenate waveforms with an optional silent gap between them."""
    if gap_ms:
        silence = np.zeros(int(_SAMPLE_RATE * gap_ms / 1000), dtype=np.int16)
        parts: list[np.ndarray] = []
        for i, a in enumerate(arrays):
            parts.append(a)
            if i < len(arrays) - 1:
                parts.append(silence)
        return np.concatenate(parts)
    return np.concatenate(arrays)


# ---------------------------------------------------------------------------
# Sound definitions
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _build_capture_sound() -> np.ndarray:
    """Two quick high-pitched blips — camera shutter feel."""
    blip = _sine(1200, 0.045, volume=0.35)
    return _concat(blip, blip, gap_ms=55)


@lru_cache(maxsize=None)
def _build_focus_sound() -> np.ndarray:
    """Ascending two-note chime — 'lock in' signal."""
    low = _sine(660, 0.12, volume=0.45)
    high = _sine(880, 0.18, volume=0.45)
    return _concat(low, high, gap_ms=30)


@lru_cache(maxsize=None)
def _build_rest_sound() -> np.ndarray:
    """Descending mellow two-note tone — 'step back' signal."""
    high = _sine(660, 0.18, volume=0.38)
    low = _sine(440, 0.22, volume=0.38)
    return _concat(high, low, gap_ms=40)


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def _play(wave: np.ndarray) -> None:
    """Play a waveform array, scaled by master volume."""
    if get_muted():
        return
    if not _init_pygame():
        return
    try:
        import pygame

        vol = get_volume()
        scaled = (wave * vol).astype(np.int16)
        sound = pygame.sndarray.make_sound(scaled)
        channel = sound.play()
        # Block until this specific channel finishes — avoids waiting on
        # unrelated sounds playing concurrently (e.g. TTS audio).
        if channel is not None:
            while channel.get_busy():
                pygame.time.wait(10)
    except Exception as exc:
        logger.warning("Failed to play sound: %s", exc)


def _play_async(wave_factory) -> None:
    """Fire-and-forget: build + play in a daemon thread."""

    def _run():
        _play(wave_factory())

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def play_notification_sync() -> None:
    """Play the notification blip synchronously (blocks until done).

    Intended for use by the TTS module on its daemon thread so a chime plays
    immediately before the spoken instruction begins.
    """
    _play(_build_capture_sound())


def play_capture() -> None:
    """Short double-blip — played when a capture cycle begins."""
    _play_async(_build_capture_sound)


def play_focus_mode() -> None:
    """Ascending chime — played when the AI switches to FOCUS mode."""
    _play_async(_build_focus_sound)


def play_rest_mode() -> None:
    """Descending tone — played when the AI switches to REST mode."""
    _play_async(_build_rest_sound)
