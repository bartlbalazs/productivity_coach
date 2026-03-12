"""Text-to-speech via Piper neural TTS (en_US-lessac-high).

Speaks coaching instructions aloud only for significant events — mode changes,
low focus redirects, posture corrections, poor break quality.

The Piper voice model (~100 MB) is downloaded once to ~/.coach/piper-voices/
and loaded into memory at application startup via ensure_voice_ready(). All
speech is fire-and-forget: speak() enqueues text and returns immediately.
A single persistent worker thread drains the queue one utterance at a time,
so multiple rapid speak() calls are played sequentially without overlap.

A short notification chime plays just before each spoken instruction so the
user has an audio cue to pay attention.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VOICE_NAME = "en_US-lessac-high"
_VOICES_DIR = Path.home() / ".coach" / "piper-voices"
# Piper synthesises at 22050 Hz mono; pygame mixer is initialised at 44100 Hz.
# We upsample via simple nearest-neighbour (np.repeat) — sufficient for speech.
_PIPER_SAMPLE_RATE = 22050
_PYGAME_SAMPLE_RATE = 44100
_UPSAMPLE_FACTOR = _PYGAME_SAMPLE_RATE // _PIPER_SAMPLE_RATE  # 2

# Maximum characters sent for synthesis. Prevents accidentally speaking multi-paragraph
# LLM dumps in the rare case of an oversized instruction field.
_MAX_SPEAK_CHARS = 400

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_voice = None  # piper.voice.PiperVoice, loaded after ensure_voice_ready()
_available: bool = False
_enabled: bool = True

# Single-worker speech queue — guarantees sequential, non-overlapping playback.
_speech_queue: queue.Queue[str] = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


def _ensure_worker() -> None:
    """Start the background speech worker thread exactly once."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker_loop, name="coach-tts-worker", daemon=True)
        t.start()
        _worker_started = True


def _worker_loop() -> None:
    """Drain the speech queue, playing one utterance at a time."""
    while True:
        text = _speech_queue.get()  # blocks until an item is available
        try:
            _speak_sync(text)
        except Exception as exc:
            logger.warning("TTS worker error: %s", exc)
        finally:
            _speech_queue.task_done()


# ---------------------------------------------------------------------------
# Public control API
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    with _lock:
        return _enabled and _available


def set_enabled(enabled: bool) -> None:
    global _enabled
    with _lock:
        _enabled = enabled


# ---------------------------------------------------------------------------
# Startup initialisation
# ---------------------------------------------------------------------------


def ensure_voice_ready() -> None:
    """Download (if needed) and load the Piper voice model.

    Call this once at application startup. It is safe to call multiple times;
    subsequent calls are instant no-ops. Blocks until the model is loaded.
    If anything fails, TTS is silently disabled.
    """
    global _voice, _available

    with _lock:
        if _available:
            return  # already loaded

        try:
            _voices_dir = _VOICES_DIR
            _voices_dir.mkdir(parents=True, exist_ok=True)

            model_path = _voices_dir / f"{_VOICE_NAME}.onnx"
            config_path = _voices_dir / f"{_VOICE_NAME}.onnx.json"

            # Download model files if missing or empty
            if (
                not model_path.exists()
                or model_path.stat().st_size == 0
                or not config_path.exists()
                or config_path.stat().st_size == 0
            ):
                logger.info(
                    "Piper voice model not found. Downloading %s to %s …"
                    " (this may take a minute on first run)",
                    _VOICE_NAME,
                    _voices_dir,
                )
                from piper.download_voices import download_voice

                download_voice(_VOICE_NAME, _voices_dir)
                logger.info("Piper voice model downloaded.")

            # Load the ONNX model into a PiperVoice.
            # If loading fails (e.g. partial download), delete and re-download once.
            logger.info("Loading Piper voice model from %s …", model_path)
            from piper.voice import PiperVoice

            try:
                _voice = PiperVoice.load(str(model_path), str(config_path))
            except Exception as load_exc:
                logger.warning(
                    "Failed to load Piper model (%s). Re-downloading …", load_exc
                )
                model_path.unlink(missing_ok=True)
                config_path.unlink(missing_ok=True)
                from piper.download_voices import download_voice

                download_voice(_VOICE_NAME, _voices_dir)
                _voice = PiperVoice.load(str(model_path), str(config_path))

            _available = True
            logger.info("Piper TTS ready (%s).", _VOICE_NAME)

        except Exception as exc:
            logger.warning(
                "Piper TTS unavailable — spoken coaching disabled. Reason: %s", exc
            )
            _available = False


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------


def speak(text: str) -> None:
    """Enqueue *text* for speech. Returns immediately. No-op if disabled."""
    if not is_enabled():
        return
    # Cap length — we don't want multi-paragraph dumps accidentally spoken.
    text = text.strip()[:_MAX_SPEAK_CHARS]
    if not text:
        return

    _ensure_worker()
    _speech_queue.put(text)


def _speak_sync(text: str) -> None:
    """Synthesise and play *text* on the calling thread (blocking)."""
    try:
        # Play notification chime first so the user knows speech is coming.
        from coach.ui.sounds import play_notification_sync

        play_notification_sync()

        # Synthesise all audio chunks and concatenate into one int16 array.
        audio_parts: list[np.ndarray] = []
        with _lock:
            voice = _voice
        if voice is None:
            return
        for chunk in voice.synthesize(text):
            audio_parts.append(chunk.audio_int16_array)

        if not audio_parts:
            return

        audio = np.concatenate(audio_parts)

        # Upsample from 22050 Hz → 44100 Hz by repeating each sample twice.
        audio_up = np.repeat(audio, _UPSAMPLE_FACTOR)

        # Play via pygame mixer (already initialised by sounds.py at 44100 Hz).
        from coach.ui.sounds import _init_pygame, get_muted, get_volume

        if get_muted():
            return
        if not _init_pygame():
            return

        import pygame

        vol = get_volume()
        scaled = (audio_up * vol).astype(np.int16)
        sound = pygame.sndarray.make_sound(scaled)
        channel = sound.play()
        if channel is not None:
            while channel.get_busy():
                pygame.time.wait(10)

    except Exception as exc:
        logger.warning("TTS playback error: %s", exc)
