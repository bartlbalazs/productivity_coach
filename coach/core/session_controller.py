"""Session lifecycle and scheduler-event business logic.

All functions here read/write ``st.session_state`` but never call any
Streamlit rendering primitives — keeping UI concerns out of this layer.
"""

from __future__ import annotations

import logging
import queue
from datetime import datetime, timedelta, timezone
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

from coach.core.agent import AnalysisResult, UserMode, extract_tasks
from coach.database import (
    CaptureRecord,
    Session,
    SessionLogEntry,
    Task,
    claim_session_scheduler,
    close_all_open_sessions,
    create_session,
    end_session,
    force_claim_session_scheduler,
    get_latest_capture_for_session,
    get_latest_closed_session,
    get_open_session_with_live_lock,
    get_or_cleanup_open_session,
    get_session_log,
    get_session_tasks,
    mark_session_stopping,
    mark_task_done,
    reopen_session,
    replace_session_tasks,
    reset_stale_stopping_sessions,
    save_session_summary,
    update_session_goal,
)
from coach.core.scheduler import (
    CycleCompleteEvent,
    CycleErrorEvent,
    CycleSkippedEvent,
    CycleStartedEvent,
    HealthCheckFailedEvent,
    MonitoringScheduler,
    SessionEndedEvent,
    SessionPausedEvent,
    SessionResumedEvent,
    SessionSummaryEvent,
)
from coach.ui.sounds import play_focus_mode, play_rest_mode


# ---------------------------------------------------------------------------
# Module-level scheduler registry
#
# Keeps a reference to the single active MonitoringScheduler for this process.
# When a Streamlit tab refreshes, its session_state is wiped but this module-
# level variable survives (same Python process), allowing the UI to re-attach
# to the already-running scheduler thread.
# ---------------------------------------------------------------------------

_active_scheduler: Optional[MonitoringScheduler] = None


def _register_scheduler(scheduler: MonitoringScheduler) -> None:
    global _active_scheduler
    _active_scheduler = scheduler


def _unregister_scheduler() -> None:
    global _active_scheduler
    _active_scheduler = None


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_cycle_complete_event(event: CycleCompleteEvent) -> None:
    """Update session state and play sounds for a completed analysis cycle."""
    prev_result = st.session_state.get("latest_result")
    st.session_state["is_analysing"] = False
    st.session_state["latest_capture"] = event.capture
    st.session_state["latest_result"] = event.result

    # Compute break quality from input snapshot.
    # The snapshot covers the interval since the *previous* check-in, so the
    # score reflects the previous mode, not the mode the LLM just decided.
    prev_mode = prev_result.mode if prev_result is not None else None
    if prev_mode == UserMode.REST and event.capture.input_snapshot is not None:
        st.session_state["latest_break_quality"] = (
            event.capture.input_snapshot.compute_break_quality()
        )
    else:
        st.session_state["latest_break_quality"] = None

    new_mode = event.result.mode
    prev_mode = prev_result.mode if prev_result is not None else None
    if new_mode != prev_mode:
        scheduler: Optional[MonitoringScheduler] = st.session_state.get("scheduler")
        if new_mode == UserMode.FOCUS:
            play_focus_mode()
            st.session_state["rest_ends_at"] = None
            if scheduler is not None:
                scheduler.set_rest_ends_at(None)
        else:
            play_rest_mode()
            completed_sprints = getattr(event, "completed_sprints", 0)
            # Enforce Pomodoro-correct minimums regardless of what the LLM suggests.
            # Every 4th sprint earns a long break (15 min); all others get 5 min.
            if completed_sprints > 0 and completed_sprints % 4 == 0:
                min_rest_secs = 900  # 15 minutes — long break
            else:
                min_rest_secs = 300  # 5 minutes — short break
            secs = max(event.result.suggested_next_interval or 0, min_rest_secs)
            rest_ends_at = datetime.now(timezone.utc) + timedelta(seconds=secs)
            st.session_state["rest_ends_at"] = rest_ends_at
            if scheduler is not None:
                scheduler.set_rest_ends_at(rest_ends_at)


def drain_event_queue() -> None:
    """Pull all pending events off the scheduler queue and update session state."""
    scheduler: Optional[MonitoringScheduler] = st.session_state.get("scheduler")
    if scheduler is None:
        return

    try:
        while True:
            event = scheduler.event_queue.get_nowait()
            st.session_state["latest_event"] = event

            if isinstance(event, CycleStartedEvent):
                logger.info("Cycle started.")
                st.session_state["is_analysing"] = True
                st.session_state["error_message"] = None
                st.session_state["skipped_message"] = None

            elif isinstance(event, CycleCompleteEvent):
                result = event.result
                logger.info(
                    "Cycle complete — mode=%s score=%s next_in=%ds.",
                    result.mode.value if result else "?",
                    result.focus_score if result else "?",
                    event.next_capture_in,
                )
                _handle_cycle_complete_event(event)

            elif isinstance(event, CycleErrorEvent):
                logger.error("Cycle error: %s", event.error)
                st.session_state["is_analysing"] = False
                st.session_state["error_message"] = event.error

            elif isinstance(event, CycleSkippedEvent):
                logger.info("Cycle skipped: %s", event.reason)
                st.session_state["is_analysing"] = False
                st.session_state["skipped_message"] = event.reason

            elif isinstance(event, HealthCheckFailedEvent):
                logger.error("Health check failed: %s", event.error)
                st.session_state["is_analysing"] = False
                st.session_state["health_check_error"] = event.error

            elif isinstance(event, SessionPausedEvent):
                logger.info("Session paused.")
                st.session_state["paused"] = True

            elif isinstance(event, SessionResumedEvent):
                logger.info("Session resumed.")
                st.session_state["paused"] = False

            elif isinstance(event, SessionSummaryEvent):
                logger.info("Session summary received: %s", event.summary.headline)
                st.session_state["session_summary"] = event.summary
                sid = st.session_state.get("session_id")
                if sid is not None:
                    logger.info("Saving session summary to DB for session %d.", sid)
                    save_session_summary(
                        sid,
                        event.summary.headline,
                        event.summary.model_dump_json(),
                    )
                    logger.info("Session summary saved for session %d.", sid)
                else:
                    logger.error(
                        "Cannot save session summary — session_id is None in session_state. "
                        "Summary headline: %s",
                        event.summary.headline,
                    )

            elif isinstance(event, SessionEndedEvent):
                logger.info("Session ended.")
                st.session_state["monitoring"] = False
                st.session_state["stopping"] = False
                st.session_state["paused"] = False
                st.session_state["scheduler"] = None
                st.session_state["is_analysing"] = False
                _unregister_scheduler()

    except queue.Empty:
        pass


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def _analysis_result_from_capture(record: CaptureRecord) -> AnalysisResult:
    """Reconstruct an AnalysisResult from a persisted CaptureRecord.

    Used when re-attaching to a running session after a page refresh so that
    the UI can show the last known analysis without waiting for the next cycle.
    """
    mode = UserMode.REST if record.is_distracted else UserMode.FOCUS
    return AnalysisResult(
        mode=mode,
        focus_score=record.focus_score,
        activity_description=record.activity_description,
        activity_label=record.activity_label or "",
        instruction=record.feedback_message,
        suggestions=record.suggestions,
        distraction_category=record.distraction_category,
        suggested_next_interval=record.suggested_next_interval,
        posture_correction=record.posture_correction,
    )


def start_monitoring() -> None:
    """Create a new session in the DB and start the scheduler.

    Any existing open session with no live scheduler lock is closed first so
    the DB does not accumulate orphaned sessions.
    """
    # Close any lingering open session before creating a fresh one.
    closed_count = close_all_open_sessions()
    if closed_count > 0:
        logger.info(
            "Closed %d leftover open session(s) before starting new one.", closed_count
        )

    goal = st.session_state.get("session_goal", "") or None
    session_id = create_session(goal=goal)
    interval_min = st.session_state["interval_min"]
    interval_max = st.session_state["interval_max"]
    logger.info(
        "Starting new session %d (goal=%r, interval=%d–%ds).",
        session_id,
        goal,
        interval_min,
        interval_max,
    )
    scheduler = MonitoringScheduler(
        session_id=session_id,
        interval_min=interval_min,
        interval_max=interval_max,
        session_goal=goal,
    )
    # Claim the DB scheduler lock before starting the thread
    claim_session_scheduler(session_id, scheduler.owner_token)
    scheduler.start()
    _register_scheduler(scheduler)

    st.session_state["scheduler"] = scheduler
    st.session_state["session_id"] = session_id
    st.session_state["session_start"] = datetime.now(timezone.utc)
    st.session_state["monitoring"] = True
    st.session_state["paused"] = False
    st.session_state["session_summary"] = None
    st.session_state["is_analysing"] = False
    st.session_state["latest_result"] = None
    st.session_state["latest_capture"] = None
    st.session_state["error_message"] = None
    st.session_state["skipped_message"] = None
    st.session_state["health_check_error"] = None
    st.session_state["session_log"] = []
    st.session_state["session_tasks"] = []
    st.session_state["current_focus"] = goal or ""


def stop_monitoring() -> None:
    """Signal the scheduler to stop and mark the session as stopping in the DB."""
    scheduler: Optional[MonitoringScheduler] = st.session_state.get("scheduler")
    sid = st.session_state.get("session_id")
    logger.info("Stop requested for session %s.", sid)
    # Mark the session as stopping in the DB *immediately* so that
    # auto_resume_if_needed() on a fresh page load won't pick it up
    # while the background thread is still winding down.
    if sid is not None:
        mark_session_stopping(sid)
    if scheduler:
        scheduler.stop()
    # Do NOT call _unregister_scheduler() here — the SessionEndedEvent handler
    # (line 194) is the single correct place to clear _active_scheduler.
    # Clearing it here races with the still-running background thread: if a
    # Streamlit rerun fires before SessionEndedEvent is processed,
    # auto_resume_if_needed() sees _active_scheduler=None, finds a live DB
    # lock, force-breaks it, and starts a second concurrent scheduler thread.
    # Give immediate UI feedback — SessionEndedEvent handler clears both flags.
    st.session_state["stopping"] = True


def resume_monitoring(
    session_id: int, goal: Optional[str], session_start: datetime
) -> None:
    """Resume an existing open session without creating a new one in the DB.

    Returns without doing anything if another scheduler instance currently
    holds the DB lock for this session (another tab is already running it).
    """
    interval_min = st.session_state["interval_min"]
    interval_max = st.session_state["interval_max"]
    logger.info(
        "Resuming session %d (goal=%r, interval=%d–%ds).",
        session_id,
        goal,
        interval_min,
        interval_max,
    )
    scheduler = MonitoringScheduler(
        session_id=session_id,
        interval_min=interval_min,
        interval_max=interval_max,
        session_goal=goal,
    )

    # Try to claim the scheduler lock; abort if another tab is already live.
    if not claim_session_scheduler(session_id, scheduler.owner_token):
        logger.warning(
            "Session %d already has a live scheduler in another tab — skipping resume.",
            session_id,
        )
        return

    scheduler.start()
    _register_scheduler(scheduler)

    log_entries: list[SessionLogEntry] = get_session_log(session_id)
    current_focus = log_entries[-1].note if log_entries else (goal or "")

    # Restore the last coaching result from DB so the main panel is not blank
    # while the user waits for the first new cycle to complete.
    latest_capture_record = get_latest_capture_for_session(session_id)
    latest_result = (
        _analysis_result_from_capture(latest_capture_record)
        if latest_capture_record is not None
        else None
    )

    st.session_state["scheduler"] = scheduler
    st.session_state["session_id"] = session_id
    st.session_state["session_start"] = session_start
    st.session_state["session_goal"] = goal or ""
    st.session_state["session_log"] = log_entries
    st.session_state["session_tasks"] = get_session_tasks(session_id)
    st.session_state["current_focus"] = current_focus
    st.session_state["monitoring"] = True
    st.session_state["paused"] = False
    st.session_state["session_summary"] = None
    st.session_state["is_analysing"] = False
    st.session_state["latest_result"] = latest_result
    st.session_state["latest_capture"] = None  # raw bytes not stored in registry
    st.session_state["latest_break_quality"] = (
        latest_capture_record.break_quality_score
        if latest_capture_record is not None
        else None
    )
    st.session_state["error_message"] = None
    st.session_state["skipped_message"] = None
    st.session_state["health_check_error"] = None

    # Sync task-list emptiness so the first cycle has the correct flag.
    scheduler.update_tasks(st.session_state["session_tasks"])


def _attach_to_running_session(
    session_id: int, goal: Optional[str], session_start: datetime
) -> None:
    """Re-connect this Streamlit tab to an already-running scheduler thread.

    Called after a page refresh when the process-level ``_active_scheduler``
    registry shows a live scheduler for the open session.  No new thread is
    started; the existing scheduler object is placed back into session_state so
    event draining and Stop/Pause controls work normally.

    The last persisted capture is loaded from the DB so the UI can show
    the most recent coaching result immediately rather than waiting for the
    next cycle.
    """
    scheduler = _active_scheduler  # guaranteed non-None by caller
    assert scheduler is not None

    log_entries: list[SessionLogEntry] = get_session_log(session_id)
    current_focus = log_entries[-1].note if log_entries else (goal or "")

    # Restore last result from DB so the coaching panel shows immediately
    latest_capture_record = get_latest_capture_for_session(session_id)
    latest_result = (
        _analysis_result_from_capture(latest_capture_record)
        if latest_capture_record is not None
        else None
    )

    st.session_state["scheduler"] = scheduler
    st.session_state["session_id"] = session_id
    st.session_state["session_start"] = session_start
    st.session_state["session_goal"] = goal or ""
    st.session_state["session_log"] = log_entries
    st.session_state["session_tasks"] = get_session_tasks(session_id)
    st.session_state["current_focus"] = current_focus
    st.session_state["monitoring"] = True
    st.session_state["paused"] = scheduler.is_paused
    st.session_state["session_summary"] = None
    # Do NOT set is_analysing=True here — the scheduler is running between cycles,
    # not actively capturing.  The next CycleStartedEvent will set it to True.
    st.session_state["is_analysing"] = False
    st.session_state["latest_result"] = latest_result
    st.session_state["latest_capture"] = None  # raw bytes not stored in registry
    st.session_state["latest_break_quality"] = (
        latest_capture_record.break_quality_score
        if latest_capture_record is not None
        else None
    )
    st.session_state["error_message"] = None
    st.session_state["skipped_message"] = None
    st.session_state["health_check_error"] = None

    # Sync task-list emptiness with the running scheduler.
    scheduler.update_tasks(st.session_state["session_tasks"])

    logger.info(
        "Re-attached UI to running scheduler for session %d after page refresh.",
        session_id,
    )


def resume_open_session() -> None:
    """Resume the most recent open session (called from the explicit Resume button).

    If the session's scheduler is already running in this process (e.g. the
    user navigated away and back), re-attaches the UI to the existing thread
    instead of starting a duplicate.  Otherwise starts a fresh scheduler thread
    via ``resume_monitoring``.
    """
    # Path 1: DB shows a live heartbeat for this session.
    live_session = get_open_session_with_live_lock()
    if live_session is not None:
        if (
            _active_scheduler is not None
            and _active_scheduler.session_id == live_session.id
            and _active_scheduler.is_running
        ):
            # Scheduler is still alive in this process — just re-attach the UI.
            logger.info(
                "Resume button: re-attaching to running scheduler for session %d.",
                live_session.id,
            )
            _attach_to_running_session(
                live_session.id, live_session.goal, live_session.start_time
            )
            return
        else:
            # Live lock in DB but no running scheduler in this process.
            # The process was restarted (hot-reload / crash) before the 30 s TTL
            # expired, so get_or_cleanup_open_session() would return None and the
            # Resume button would silently do nothing.  Force-break the stale lock
            # so claim_session_scheduler() will succeed inside resume_monitoring.
            logger.info(
                "Resume button: process restarted, breaking stale lock for session %d "
                "and starting fresh scheduler.",
                live_session.id,
            )
            force_claim_session_scheduler(live_session.id)
            resume_monitoring(
                live_session.id, live_session.goal, live_session.start_time
            )
            return

    # Path 2: no live lock at all — start a fresh scheduler.
    open_session = get_or_cleanup_open_session()
    if open_session is None:
        logger.info("Resume button: no resumable session found.")
        return
    logger.info(
        "Resume button: starting fresh scheduler for session %d.", open_session.id
    )
    resume_monitoring(open_session.id, open_session.goal, open_session.start_time)


def resume_latest_session() -> None:
    """Reopen the most recently stopped session and resume monitoring it.

    Clears the session's ``end_time`` (and summary) in the DB so it is treated
    as an open session again, then hands off to ``resume_monitoring`` which
    starts a fresh scheduler thread and restores all UI state (log, tasks,
    last coaching result).

    Does nothing if no closed session exists.
    """
    session = get_latest_closed_session()
    if session is None:
        logger.info("resume_latest_session: no closed session found.")
        return
    logger.info(
        "Reopening stopped session %d (goal=%r) for resume.", session.id, session.goal
    )
    reopen_session(session.id)
    resume_monitoring(session.id, session.goal, session.start_time)


def auto_resume_if_needed() -> None:
    """On fresh browser connection, resume the most recent open session if one exists.

    Two paths:
    1. Live lock (fresh heartbeat) + process-level registry has a scheduler →
       re-attach the UI to the running thread (page refresh scenario).
    2. No live lock (stale/NULL heartbeat) → start a fresh scheduler thread
       (crash-recovery scenario).
    """
    if st.session_state.get("monitoring"):
        return  # already running — normal rerun within the same tab

    # Recover sessions that were left with stopping=1 after a crash (i.e. the
    # process died after mark_session_stopping() but before end_session()).
    # Their scheduler heartbeat will have expired, so we clear the flag to make
    # them visible to get_or_cleanup_open_session() below.
    recovered = reset_stale_stopping_sessions()
    if recovered:
        logger.info(
            "Reset stopping=0 for %d session(s) orphaned mid-shutdown.", recovered
        )

    # --- Path 1: page refresh — scheduler still running in this process ---
    live_session = get_open_session_with_live_lock()
    if live_session is not None:
        if (
            _active_scheduler is not None
            and _active_scheduler.session_id == live_session.id
            and _active_scheduler.is_running
        ):
            logger.info(
                "Auto-resume: re-attaching to running scheduler for session %d (page refresh).",
                live_session.id,
            )
            _attach_to_running_session(
                live_session.id, live_session.goal, live_session.start_time
            )
            return
        else:
            # Live lock in DB but no running scheduler in this process — the
            # process was restarted mid-session (e.g. Streamlit hot-reload or
            # crash) before the old lock expired (TTL = 30 s).  The previous
            # owner is definitively gone, so force-break the stale lock before
            # resuming; without this, claim_session_scheduler would fail and
            # the session would silently remain idle until the TTL expires.
            logger.info(
                "Auto-resume: process restarted, breaking stale lock for session %d "
                "and starting fresh scheduler.",
                live_session.id,
            )
            # Clear the stale lock so claim_session_scheduler inside
            # resume_monitoring can acquire it unconditionally.
            force_claim_session_scheduler(live_session.id)
            resume_monitoring(
                live_session.id, live_session.goal, live_session.start_time
            )
            return

    # --- Path 2: crash recovery — start a fresh scheduler ---
    open_session = get_or_cleanup_open_session()
    if open_session is None:
        return  # no open session — idle start
    logger.info(
        "Auto-resume: crash-recovery resume for session %d (stale lock).",
        open_session.id,
    )
    resume_monitoring(open_session.id, open_session.goal, open_session.start_time)


# ---------------------------------------------------------------------------
# Session log mutations (shared between UI layer and session controller)
# ---------------------------------------------------------------------------


def submit_log_entry(note: str) -> None:
    """Persist a new log entry to the DB, update state, and sync the scheduler."""
    from coach.database import add_session_log_entry

    session_id: Optional[int] = st.session_state.get("session_id")
    scheduler: Optional[MonitoringScheduler] = st.session_state.get("scheduler")
    log_entries: list[SessionLogEntry] = st.session_state.get("session_log", [])

    logger.info("Log entry submitted for session %s: %r", session_id, note)

    if session_id:
        entry_id = add_session_log_entry(session_id, note)
        new_entry = SessionLogEntry(
            id=entry_id,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            note=note,
        )
        updated_log = log_entries + [new_entry]
        st.session_state["session_log"] = updated_log

        # Extract tasks from the full updated log via LLM, excluding already-done tasks
        try:
            current_tasks: list[Task] = st.session_state.get("session_tasks", [])
            had_active_tasks = any(not t.done for t in current_tasks)
            done_texts = [t.text for t in current_tasks if t.done]
            logger.info(
                "Extracting tasks from log (%d entries, %d done).",
                len(updated_log),
                len(done_texts),
            )
            new_texts = extract_tasks(updated_log, done_task_texts=done_texts)
            tasks = replace_session_tasks(session_id, new_texts)
            logger.info("Task extraction complete: %d task(s).", len(tasks))
            st.session_state["session_tasks"] = tasks
            if scheduler:
                scheduler.update_tasks(tasks)
            # If there were no active tasks before this submission, auto-focus the
            # first newly extracted task so the user immediately has a task in focus.
            if not had_active_tasks:
                new_active = [t for t in tasks if not t.done]
                if new_active:
                    logger.info(
                        "No prior active tasks — auto-focusing first extracted task: %r",
                        new_active[0].text,
                    )
                    # Override the note-based focus with the cleaner task text.
                    st.session_state["current_focus"] = new_active[0].text
                    st.session_state["session_goal"] = new_active[0].text
                    if scheduler:
                        scheduler.update_goal(new_active[0].text)
                    if session_id:
                        update_session_goal(session_id, new_active[0].text)
                    return  # focus and goal already set; skip the block below
        except Exception as exc:
            logger.warning("Task extraction failed, keeping existing tasks: %s", exc)

    st.session_state["current_focus"] = note
    st.session_state["session_goal"] = note

    if scheduler:
        scheduler.update_goal(note)
    if session_id:
        update_session_goal(session_id, note)


def set_current_focus(text: str) -> None:
    """Set the current focus text without creating a log entry or extracting tasks.

    Updates the in-session display, notifies the scheduler so future LLM cycles
    use the new goal as context, and persists the goal to the DB so it survives
    a page refresh.
    """
    session_id: Optional[int] = st.session_state.get("session_id")
    scheduler: Optional[MonitoringScheduler] = st.session_state.get("scheduler")

    st.session_state["current_focus"] = text
    st.session_state["session_goal"] = text

    if scheduler:
        scheduler.update_goal(text)
    if session_id:
        update_session_goal(session_id, text)
        logger.info("Current focus updated for session %d: %r", session_id, text)


def finish_task(task_id: int) -> None:
    """Mark a task as done in the DB and update session state."""
    logger.info("Marking task %d as done.", task_id)
    mark_task_done(task_id)

    tasks: list[Task] = st.session_state.get("session_tasks", [])
    updated: list[Task] = []
    for t in tasks:
        if t.id == task_id:
            updated.append(
                Task(
                    id=t.id,
                    session_id=t.session_id,
                    created_at=t.created_at,
                    text=t.text,
                    done=True,
                )
            )
        else:
            updated.append(t)
    st.session_state["session_tasks"] = updated

    scheduler: Optional[MonitoringScheduler] = st.session_state.get("scheduler")
    if scheduler:
        scheduler.update_tasks(updated)
