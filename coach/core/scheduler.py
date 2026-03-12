"""Random-interval scheduler that drives capture→analyse→persist cycles."""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from coach.core.agent import (
    AnalysisResult,
    CycleState,
    SessionSummary,
    UserMode,
    count_completed_sprints,
    generate_summary,
    health_check,
    run_cycle,
)
from coach.core.capture import CaptureResult
from coach.config import config
from coach.database import (
    end_session,
    refresh_scheduler_heartbeat,
    release_session_scheduler,
)
from coach.integrations.input_monitor import InputMonitor
from coach.integrations.notify import send_native
from coach.integrations import tts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events pushed onto the result queue for the UI to consume
# ---------------------------------------------------------------------------


@dataclass
class CycleStartedEvent:
    timestamp: datetime


@dataclass
class CycleCompleteEvent:
    timestamp: datetime
    capture: CaptureResult
    result: AnalysisResult
    record_id: int
    next_capture_in: int  # seconds
    completed_sprints: int = 0  # number of FOCUS sprints completed this session


@dataclass
class CycleErrorEvent:
    timestamp: datetime
    error: str
    next_capture_in: int  # seconds


@dataclass
class CycleSkippedEvent:
    timestamp: datetime
    reason: str
    next_capture_in: int  # seconds


@dataclass
class HealthCheckFailedEvent:
    timestamp: datetime
    error: str


@dataclass
class SessionPausedEvent:
    timestamp: datetime


@dataclass
class SessionResumedEvent:
    timestamp: datetime


@dataclass
class SessionSummaryEvent:
    timestamp: datetime
    summary: SessionSummary


@dataclass
class SessionEndedEvent:
    timestamp: datetime


SchedulerEvent = (
    CycleStartedEvent
    | CycleCompleteEvent
    | CycleErrorEvent
    | CycleSkippedEvent
    | HealthCheckFailedEvent
    | SessionPausedEvent
    | SessionResumedEvent
    | SessionSummaryEvent
    | SessionEndedEvent
)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class MonitoringScheduler:
    """
    Runs capture→analyse→persist cycles at random intervals in a background thread.

    Each cycle is orchestrated entirely by the LangGraph graph in agent.py.
    The scheduler's only jobs are:
      - deciding *when* to trigger the next cycle
      - pushing SchedulerEvents onto a queue for the Streamlit UI to poll
    """

    def __init__(
        self,
        session_id: int,
        interval_min: Optional[int] = None,
        interval_max: Optional[int] = None,
        session_goal: Optional[str] = None,
        on_event: Optional[Callable[[SchedulerEvent], None]] = None,
        owner_token: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.interval_min = interval_min or config.capture_interval_min
        self.interval_max = interval_max or config.capture_interval_max
        self.session_goal = session_goal
        self.on_event = on_event
        # Unique token identifying this scheduler instance as the DB lock holder
        self.owner_token: str = owner_token or str(uuid.uuid4())

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = paused
        self._thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self.event_queue: queue.Queue[SchedulerEvent] = queue.Queue()
        self._next_capture_at: Optional[datetime] = None
        # Adaptive interval: set by the last cycle's suggested_next_interval
        self._adaptive_interval: Optional[int] = None
        # Track previous mode for mode-change notifications
        self._prev_mode: Optional[UserMode] = None
        # Track when a persistent issue was first seen (for 3-minute speak gate)
        self._low_focus_since: Optional[datetime] = None
        self._posture_issue_since: Optional[datetime] = None
        # Input monitor — started with the session, stopped when it ends
        self._input_monitor: InputMonitor = InputMonitor()
        # Set to True while a capture cycle is actively in-flight so the UI
        # can detect a missed CycleCompleteEvent and recover.
        self._cycle_active: bool = False
        # True when no active tasks are defined — injected into each cycle so the
        # LLM can instruct the user to fill the task list.
        self._tasks_empty: bool = True

    # ------------------------------------------------------------------
    # Public control interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Scheduler is already running.")
            return
        self._stop_event.clear()
        self._input_monitor.start()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="coach-scheduler",
            daemon=True,
        )
        self._thread.start()
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat,
            name="coach-scheduler-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        logger.info("Monitoring scheduler started (session %d).", self.session_id)

    def stop(self) -> None:
        """Signal the background thread to stop after the current sleep."""
        self._stop_event.set()
        self._pause_event.clear()  # unblock pause if currently paused
        # Release the DB lock immediately so other tabs can take over if needed
        try:
            release_session_scheduler(self.session_id, self.owner_token)
        except Exception:
            logger.exception("Failed to release scheduler lock on stop.")
        logger.info("Stop signal sent to scheduler.")

    def update_goal(self, goal: str) -> None:
        """Update the session goal used by future cycles."""
        self.session_goal = goal or None

    def update_tasks(self, tasks: list) -> None:
        """Update the task-list emptiness flag used by future cycles.

        Pass the current list[Task] (or any list); the scheduler only needs to
        know whether any active (not-done) tasks exist.
        """
        active = [t for t in tasks if not getattr(t, "done", False)]
        self._tasks_empty = len(active) == 0

    def pause(self) -> None:
        """Pause scheduling after the current cycle completes."""
        if not self._pause_event.is_set():
            self._pause_event.set()
            self._next_capture_at = None
            self._push(SessionPausedEvent(timestamp=datetime.now(timezone.utc)))
            logger.info("Scheduler paused (session %d).", self.session_id)

    def resume(self) -> None:
        """Resume scheduling from a paused state."""
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._push(SessionResumedEvent(timestamp=datetime.now(timezone.utc)))
            logger.info("Scheduler resumed (session %d).", self.session_id)

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set() and not self._stop_event.is_set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_cycle_running(self) -> bool:
        """True while a capture→analyse→persist cycle is actively in flight.

        Distinct from is_running (the thread is alive) — this is only True
        during the actual LLM call, not during the inter-cycle sleep.
        Used by the UI to detect a missed CycleCompleteEvent and recover.
        """
        return self._cycle_active

    @property
    def next_capture_at(self) -> Optional[datetime]:
        return self._next_capture_at

    @property
    def seconds_until_next(self) -> Optional[int]:
        if self._next_capture_at is None:
            return None
        remaining = (self._next_capture_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _HEARTBEAT_INTERVAL_SECS: int = 10  # refresh every 10 s (TTL is 30 s)

    def _run_heartbeat(self) -> None:
        """Periodically refresh the scheduler heartbeat in the DB.

        If the heartbeat refresh fails (e.g. session was closed by another process),
        this thread signals the scheduler to stop.
        """
        while not self._stop_event.is_set():
            try:
                # If refresh returns False, our lock is gone or the session ended.
                if not refresh_scheduler_heartbeat(self.session_id, self.owner_token):
                    logger.warning(
                        "Scheduler heartbeat failed (session %d closed or lock lost). Stopping.",
                        self.session_id,
                    )
                    self.stop()
                    return
            except Exception:
                logger.exception("Failed to refresh scheduler heartbeat.")
            self._stop_event.wait(timeout=self._HEARTBEAT_INTERVAL_SECS)

    def _random_interval(self) -> int:
        return random.randint(self.interval_min, self.interval_max)

    def _push(self, event: SchedulerEvent) -> None:
        self.event_queue.put(event)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                logger.exception("on_event callback raised an exception.")

    def _sleep_interruptible(self, seconds: int) -> None:
        """Sleep for *seconds* but wake immediately if stop or pause is requested."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stop_event.is_set() or self._pause_event.is_set():
                return
            time.sleep(1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """
        Continuously trigger cycles until stopped.

        The first cycle runs immediately; subsequent cycles are separated by
        a random interval between interval_min and interval_max seconds.
        Honours pause: while paused, the loop blocks in 1-second ticks.
        """
        # --- health check before anything else ---
        try:
            health_check()
        except Exception as exc:
            logger.error("Health check failed: %s", exc)
            self._push(
                HealthCheckFailedEvent(
                    timestamp=datetime.now(timezone.utc),
                    error=str(exc),
                )
            )
            self._on_session_ended()
            return

        first_cycle = True

        while not self._stop_event.is_set():
            # Block while paused (without consuming CPU)
            while self._pause_event.is_set() and not self._stop_event.is_set():
                time.sleep(1)

            if self._stop_event.is_set():
                break

            if not first_cycle:
                # Use AI-suggested interval if available, else random
                if self._adaptive_interval is not None:
                    wait_secs = self._adaptive_interval
                    self._adaptive_interval = None
                    logger.info("Next cycle in %d seconds (adaptive).", wait_secs)
                else:
                    wait_secs = self._random_interval()
                    logger.info("Next cycle in %d seconds.", wait_secs)
                self._next_capture_at = datetime.now(timezone.utc) + timedelta(
                    seconds=wait_secs
                )
                self._sleep_interruptible(wait_secs)

            first_cycle = False

            if self._stop_event.is_set() or self._pause_event.is_set():
                continue

            self._next_capture_at = None
            self._run_cycle()

        self._on_session_ended()

    def _run_cycle(self) -> None:
        """
        Trigger one capture→analyse→persist cycle via the LangGraph graph
        and push the appropriate event onto the queue.
        """
        self._cycle_active = True
        self._push(CycleStartedEvent(timestamp=datetime.now(timezone.utc)))

        try:
            final_state: CycleState = run_cycle(
                self.session_id,
                self.session_goal,
                self._input_monitor,
                tasks_empty=self._tasks_empty,
            )

            result = final_state["result"]

            # Use AI-suggested interval for the next cycle if provided and in bounds
            suggested = result.suggested_next_interval if result else None  # type: ignore[union-attr]
            if suggested is not None:
                clamped = max(self.interval_min, min(self.interval_max, suggested))
                self._adaptive_interval = clamped

            # Estimate the upcoming wait for the event payload only.
            next_wait = self._adaptive_interval or self._random_interval()

            self._push(
                CycleCompleteEvent(
                    timestamp=final_state["timestamp"],  # type: ignore[arg-type]
                    capture=final_state["capture"],  # type: ignore[arg-type]
                    result=final_state["result"],  # type: ignore[arg-type]
                    record_id=final_state["record_id"],  # type: ignore[arg-type]
                    next_capture_in=next_wait,
                    completed_sprints=final_state.get("completed_sprints", 0),  # type: ignore[union-attr]
                )
            )

            # --- Native desktop notifications ---
            capture = final_state.get("capture")
            self._fire_cycle_notifications(result, capture)

        except Exception as exc:
            logger.exception("Cycle failed.")
            next_wait = self._random_interval()
            self._push(
                CycleErrorEvent(
                    timestamp=datetime.now(timezone.utc),
                    error=str(exc),
                    next_capture_in=next_wait,
                )
            )
        finally:
            self._cycle_active = False

    # Minimum duration an issue must persist before the user is notified/spoken to.
    _ISSUE_SPEAK_THRESHOLD_SECS: int = 3 * 60  # 3 minutes

    def _fire_cycle_notifications(
        self,
        result: Optional[AnalysisResult],
        capture: Optional[CaptureResult] = None,
    ) -> None:
        """Fire native notify-send notifications for significant coaching events.

        Posture and low-focus alerts are only surfaced (notification + TTS) once
        the issue has persisted for at least _ISSUE_SPEAK_THRESHOLD_SECS seconds
        of consecutive captures.  The first occurrence is always silent.
        """
        if result is None:
            return
        now = datetime.now(timezone.utc)
        current_mode = result.mode

        # Mode change
        if self._prev_mode is not None and current_mode != self._prev_mode:
            if current_mode == UserMode.FOCUS:
                send_native("FOCUS Mode", result.instruction, "mode_change")
                tts.speak(f"Focus. {result.instruction}")
            else:
                send_native("REST Mode", result.instruction, "mode_change")
                tts.speak(f"Rest time. {result.instruction}")
                # Reset input counters so the first REST cycle's snapshot only
                # covers activity during the REST period, not the preceding FOCUS work.
                self._input_monitor.snapshot_and_reset()
            # Reset persistence counters on any mode change
            self._low_focus_since = None
            self._posture_issue_since = None

        # Low focus (only in FOCUS mode to avoid noise during REST)
        if current_mode == UserMode.FOCUS and result.focus_score <= 4:
            if self._low_focus_since is None:
                self._low_focus_since = now
                logger.debug("Low-focus issue started tracking at %s.", now)
            elapsed = (now - self._low_focus_since).total_seconds()
            if elapsed >= self._ISSUE_SPEAK_THRESHOLD_SECS:
                send_native(
                    "Low Focus Detected",
                    result.instruction,
                    "low_focus",
                )
                tts.speak(result.instruction)
        else:
            self._low_focus_since = None

        # Posture correction
        if result.posture_correction:
            if self._posture_issue_since is None:
                self._posture_issue_since = now
                logger.debug("Posture issue started tracking at %s.", now)
            elapsed = (now - self._posture_issue_since).total_seconds()
            if elapsed >= self._ISSUE_SPEAK_THRESHOLD_SECS:
                send_native(
                    "Posture Correction",
                    result.posture_correction[:100],  # cap length for notification
                    "posture",
                )
                tts.speak(result.posture_correction)
        else:
            self._posture_issue_since = None

        # Poor break quality (REST mode only) — computed from input metrics
        if current_mode == UserMode.REST and capture is not None:
            snapshot = capture.input_snapshot
            if snapshot is not None:
                bq = snapshot.compute_break_quality()
                if bq <= 3:
                    send_native(
                        "Poor Break Quality",
                        result.instruction,
                        "break_quality",
                    )
                    tts.speak(result.instruction)

        self._prev_mode = current_mode

    def _on_session_ended(self) -> None:
        self._input_monitor.stop()
        try:
            end_session(self.session_id)
        except Exception:
            logger.exception("Failed to mark session as ended in DB.")

        try:
            summary = generate_summary(self.session_id)
            self._push(
                SessionSummaryEvent(
                    timestamp=datetime.now(timezone.utc),
                    summary=summary,
                )
            )
        except Exception:
            logger.exception("Failed to generate session summary.")

        self._push(SessionEndedEvent(timestamp=datetime.now(timezone.utc)))
        logger.info("Monitoring scheduler stopped (session %d).", self.session_id)
