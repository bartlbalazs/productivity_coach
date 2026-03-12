"""SQLite persistence layer for the Productivity Coach."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Generator, Optional

from coach.config import config


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class LlmCallRecord:
    id: int
    session_id: Optional[int]
    timestamp: datetime
    call_type: str  # health_check | analyse | session_summary | weekly_summary
    model: str
    request_text: str  # serialised messages, image bytes replaced with placeholders
    response_text: Optional[str]
    token_input: Optional[int]
    token_output: Optional[int]
    latency_ms: Optional[int]
    error: Optional[str]


@dataclass
class Session:
    id: int
    start_time: datetime
    end_time: Optional[datetime]
    goal: Optional[str] = None
    summary: Optional[str] = None
    summary_json: Optional[str] = None


@dataclass
class SessionLogEntry:
    id: int
    session_id: int
    timestamp: datetime
    note: str


@dataclass
class Task:
    id: int
    session_id: int
    created_at: datetime
    text: str
    done: bool = False


@dataclass
class CaptureRecord:
    id: int
    session_id: int
    timestamp: datetime
    focus_score: int  # 1-10
    is_distracted: bool
    activity_description: str
    feedback_message: str
    suggestions: list[str]
    activity_label: Optional[str] = None
    distraction_category: Optional[str] = None
    suggested_next_interval: Optional[int] = None
    break_quality_score: Optional[int] = None
    posture_correction: Optional[str] = None
    # Input activity metrics
    keystroke_count: Optional[int] = None
    mouse_distance_px: Optional[float] = None
    click_count: Optional[int] = None
    # Fitbit health metrics (optional — only present when Fitbit is connected)
    heart_rate: Optional[int] = None
    resting_hr: Optional[int] = None
    hrv: Optional[float] = None
    steps: Optional[int] = None
    # Raw image is stored but not always loaded — use load_images=True
    webcam_image: Optional[bytes] = None

    @property
    def mode_label(self) -> str:
        """Return 'REST' or 'FOCUS' based on is_distracted."""
        return "REST" if self.is_distracted else "FOCUS"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: str) -> datetime:
    """Parse an ISO datetime string and ensure it is UTC-aware.

    Older rows may have been stored without timezone info (naive UTC).
    Newer rows include the +00:00 suffix. Either way we return an aware datetime.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _ensure_db_dir() -> None:
    if d := os.path.dirname(config.db_path):
        os.makedirs(d, exist_ok=True)


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(config.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time  TEXT NOT NULL,
    end_time    TEXT,
    goal        TEXT
);

CREATE TABLE IF NOT EXISTS captures (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              INTEGER NOT NULL REFERENCES sessions(id),
    timestamp               TEXT NOT NULL,
    focus_score             INTEGER NOT NULL,
    is_distracted           INTEGER NOT NULL,
    activity_description    TEXT NOT NULL,
    feedback_message        TEXT NOT NULL,
    suggestions             TEXT NOT NULL,
    webcam_image            BLOB,
    activity_label          TEXT,
    distraction_category    TEXT,
    suggested_next_interval INTEGER,
    break_quality_score     INTEGER,
    posture_correction      TEXT,
    keystroke_count         INTEGER,
    mouse_distance_px       REAL,
    click_count             INTEGER,
    heart_rate              INTEGER,
    resting_hr              INTEGER,
    hrv                     REAL,
    steps                   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_captures_session
    ON captures(session_id, timestamp);

CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES sessions(id),
    timestamp       TEXT NOT NULL,
    call_type       TEXT NOT NULL,
    model           TEXT NOT NULL,
    request_text    TEXT NOT NULL,
    response_text   TEXT,
    token_input     INTEGER,
    token_output    INTEGER,
    latency_ms      INTEGER,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_session
    ON llm_calls(session_id, timestamp);

CREATE TABLE IF NOT EXISTS session_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    timestamp   TEXT NOT NULL,
    note        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_log_session
    ON session_log(session_id, timestamp);

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    created_at  TEXT NOT NULL,
    text        TEXT NOT NULL,
    done        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tasks_session
    ON tasks(session_id, created_at);
"""

# Migrations: columns added after the initial schema
_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN goal TEXT",
    "ALTER TABLE captures ADD COLUMN activity_label TEXT",
    "ALTER TABLE captures ADD COLUMN distraction_category TEXT",
    "ALTER TABLE captures ADD COLUMN suggested_next_interval INTEGER",
    "ALTER TABLE captures ADD COLUMN break_quality_score INTEGER",
    "ALTER TABLE captures ADD COLUMN posture_correction TEXT",
    "ALTER TABLE captures ADD COLUMN keystroke_count INTEGER",
    "ALTER TABLE captures ADD COLUMN mouse_distance_px REAL",
    "ALTER TABLE captures ADD COLUMN click_count INTEGER",
    "ALTER TABLE sessions ADD COLUMN summary TEXT",
    "ALTER TABLE sessions ADD COLUMN summary_json TEXT",
    "ALTER TABLE captures ADD COLUMN heart_rate INTEGER",
    "ALTER TABLE captures ADD COLUMN resting_hr INTEGER",
    "ALTER TABLE captures ADD COLUMN hrv REAL",
    "ALTER TABLE captures ADD COLUMN steps INTEGER",
    "ALTER TABLE sessions ADD COLUMN stopping INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN scheduler_owner TEXT DEFAULT NULL",
    "ALTER TABLE sessions ADD COLUMN scheduler_heartbeat TEXT DEFAULT NULL",
    "ALTER TABLE session_log ADD COLUMN done INTEGER NOT NULL DEFAULT 0",
]

# Seconds after which a scheduler heartbeat is considered stale
_SCHEDULER_LOCK_TTL_SECS: int = 30


def init_db() -> None:
    """Initialise the database schema (idempotent) and run any pending migrations."""
    _ensure_db_dir()
    with _get_conn() as conn:
        conn.executescript(_SCHEMA)
        # Run each migration; silently skip if the column already exists
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists


# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------


def create_session(goal: Optional[str] = None) -> int:
    """Create a new monitoring session and return its ID."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (start_time, goal) VALUES (?, ?)",
            (now, goal),
        )
        return cur.lastrowid or 0  # type: ignore[return-value]


def update_session_goal(session_id: int, goal: str) -> None:
    """Update the goal for an active session."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET goal = ? WHERE id = ?",
            (goal, session_id),
        )


def save_session_summary(session_id: int, headline: str, summary_json: str) -> None:
    """Persist the session summary headline and full JSON for a completed session."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET summary = ?, summary_json = ? WHERE id = ?",
            (headline, summary_json, session_id),
        )


def get_session_summary_json(session_id: int) -> Optional[str]:
    """Return the raw summary_json string for a session, or None if not available."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT summary_json FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return row["summary_json"] if row else None


def end_session(session_id: int) -> None:
    """Mark a session as ended (also clears the stopping flag)."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET end_time = ?, stopping = 0 WHERE id = ?",
            (now, session_id),
        )


def mark_session_stopping(session_id: int) -> None:
    """Flag a session as stopping so auto-resume will not pick it up."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET stopping = 1 WHERE id = ?",
            (session_id,),
        )


def claim_session_scheduler(session_id: int, owner_token: str) -> bool:
    """Atomically claim the scheduler lock for *session_id*.

    Returns True if the lock was acquired (i.e. no live owner exists),
    False if another scheduler is already holding a fresh lock.

    A lock is considered stale when ``scheduler_heartbeat`` has not been
    refreshed for more than ``_SCHEDULER_LOCK_TTL_SECS`` seconds.
    """
    now = datetime.now(timezone.utc)
    threshold = (now - timedelta(seconds=_SCHEDULER_LOCK_TTL_SECS)).isoformat()

    with _get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE sessions
               SET scheduler_owner = ?,
                   scheduler_heartbeat = ?
             WHERE id = ?
               AND (
                   scheduler_owner IS NULL
                   OR scheduler_heartbeat IS NULL
                   OR scheduler_heartbeat < ?
               )
            """,
            (owner_token, now.isoformat(), session_id, threshold),
        )
        return cur.rowcount == 1


def refresh_scheduler_heartbeat(session_id: int, owner_token: str) -> bool:
    """Update the heartbeat timestamp for an active scheduler lock.

    Returns True if the lock was refreshed, False if the lock is invalid
    or the session has been closed (end_time is not NULL).
    """
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE sessions
               SET scheduler_heartbeat = ?
             WHERE id = ?
               AND scheduler_owner = ?
               AND end_time IS NULL
            """,
            (now, session_id, owner_token),
        )
        return cur.rowcount == 1


def release_session_scheduler(session_id: int, owner_token: str) -> None:
    """Release the scheduler lock so another process/tab can claim it."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET scheduler_owner = NULL, scheduler_heartbeat = NULL "
            "WHERE id = ? AND scheduler_owner = ?",
            (session_id, owner_token),
        )


def force_claim_session_scheduler(session_id: int) -> None:
    """Unconditionally claim the scheduler lock for *session_id*.

    Unlike ``claim_session_scheduler``, this ignores the TTL check and
    overwrites any existing owner.  Use only when it is certain that the
    previous lock holder is dead — specifically when the DB shows a live lock
    but no scheduler thread exists in the current process (i.e. the process
    was restarted and the old lock has not yet expired).

    We clear both fields to NULL so that the immediately-following call to
    ``claim_session_scheduler`` (inside ``resume_monitoring``) always succeeds.
    Writing a fresh heartbeat here would block that subsequent claim because
    ``claim_session_scheduler`` only grants the lock when the heartbeat is stale
    or NULL.
    """
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET scheduler_owner = NULL, scheduler_heartbeat = NULL WHERE id = ?",
            (session_id,),
        )


def close_all_open_sessions() -> int:
    """Close ALL sessions that are currently open, regardless of lock status.

    Returns the number of sessions closed. Used to enforce a strict singleton
    session policy when starting a new one.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "UPDATE sessions SET end_time = ?, stopping = 0 WHERE end_time IS NULL",
            (now,),
        )
        return cur.rowcount


def get_or_cleanup_open_session() -> Optional[Session]:
    """Return the most recent open (unended, non-stopping) session with no live
    scheduler owner, or None.

    A session is considered to have a live owner when its ``scheduler_heartbeat``
    was refreshed within the last ``_SCHEDULER_LOCK_TTL_SECS`` seconds.  Sessions
    whose heartbeat is stale (or NULL) are eligible for resume.

    If multiple eligible sessions exist (e.g. from a previous crash), all but
    the most recent are closed as a cleanup side effect.
    """
    threshold = (
        datetime.now(timezone.utc) - timedelta(seconds=_SCHEDULER_LOCK_TTL_SECS)
    ).isoformat()

    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, start_time, end_time, goal FROM sessions
             WHERE end_time IS NULL
               AND stopping = 0
               AND (
                   scheduler_owner IS NULL
                   OR scheduler_heartbeat IS NULL
                   OR scheduler_heartbeat < ?
               )
             ORDER BY start_time DESC
            """,
            (threshold,),
        ).fetchall()

    if not rows:
        return None

    # Close any extras beyond the most recent
    for row in rows[1:]:
        end_session(row["id"])

    r = rows[0]
    return Session(
        id=r["id"],
        start_time=_parse_dt(r["start_time"]),
        end_time=None,
        goal=r["goal"],
    )


def get_open_session_with_live_lock() -> Optional[Session]:
    """Return the most recent open session that currently holds a live scheduler lock.

    This is the complement of ``get_or_cleanup_open_session``: it returns only
    sessions whose heartbeat is *fresh* (within ``_SCHEDULER_LOCK_TTL_SECS``),
    meaning a scheduler thread is actively running for that session in this
    process.  Used after a window refresh to detect and re-attach to the
    running scheduler rather than starting a new one.
    """
    threshold = (
        datetime.now(timezone.utc) - timedelta(seconds=_SCHEDULER_LOCK_TTL_SECS)
    ).isoformat()

    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, start_time, end_time, goal FROM sessions
             WHERE end_time IS NULL
               AND stopping = 0
               AND scheduler_owner IS NOT NULL
               AND scheduler_heartbeat IS NOT NULL
               AND scheduler_heartbeat >= ?
             ORDER BY start_time DESC
             LIMIT 1
            """,
            (threshold,),
        ).fetchone()

    if row is None:
        return None
    return Session(
        id=row["id"],
        start_time=_parse_dt(row["start_time"]),
        end_time=None,
        goal=row["goal"],
    )


def get_latest_capture_for_session(session_id: int) -> Optional[CaptureRecord]:
    """Return the single most recent capture for a session, or None if none exist."""
    results = get_recent_captures(session_id, limit=1)
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Session log operations
# ---------------------------------------------------------------------------


def add_session_log_entry(session_id: int, note: str) -> int:
    """Append a free-form note to the session log and return the new entry ID."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO session_log (session_id, timestamp, note) VALUES (?, ?, ?)",
            (session_id, now, note),
        )
        return cur.lastrowid or 0  # type: ignore[return-value]


def get_session_log(session_id: int) -> list[SessionLogEntry]:
    """Return all log entries for a session in chronological order (oldest first)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, session_id, timestamp, note FROM session_log "
            "WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
    return [
        SessionLogEntry(
            id=r["id"],
            session_id=r["session_id"],
            timestamp=_parse_dt(r["timestamp"]),
            note=r["note"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Task operations
# ---------------------------------------------------------------------------


def get_session_tasks(session_id: int) -> list[Task]:
    """Return all tasks for a session ordered by creation time (oldest first)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, session_id, created_at, text, done FROM tasks "
            "WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return [
        Task(
            id=r["id"],
            session_id=r["session_id"],
            created_at=_parse_dt(r["created_at"]),
            text=r["text"],
            done=bool(r["done"]),
        )
        for r in rows
    ]


def replace_session_tasks(session_id: int, texts: list[str]) -> list[Task]:
    """Replace all non-done tasks with a fresh list extracted by the LLM.

    Completed tasks are preserved unchanged.  Active tasks are deleted and
    replaced with the new ``texts`` list.  Returns the full updated task list
    (done tasks first in original order, then new active tasks).
    """
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM tasks WHERE session_id = ? AND done = 0",
            (session_id,),
        )
        for text in texts:
            conn.execute(
                "INSERT INTO tasks (session_id, created_at, text, done) VALUES (?, ?, ?, 0)",
                (session_id, now, text),
            )
    return get_session_tasks(session_id)


def mark_task_done(task_id: int) -> None:
    """Mark a task as completed."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET done = 1 WHERE id = ?",
            (task_id,),
        )


def get_mode_streak_start(session_id: int) -> Optional[tuple[str, datetime]]:
    """Return (mode_label, streak_start) for the current unbroken mode streak.

    Walks backwards through every capture for the session (lightweight query —
    only timestamp and is_distracted are fetched) until the mode changes.
    Returns None if no captures exist for the session.

    This is the source of truth for streak duration — it is not capped by the
    LLM context window (history_context_size), so even a 40-minute unbroken
    FOCUS session will be reported correctly.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT timestamp, is_distracted FROM captures "
            "WHERE session_id = ? ORDER BY timestamp DESC",
            (session_id,),
        ).fetchall()

    if not rows:
        return None

    current_mode = "REST" if rows[0]["is_distracted"] else "FOCUS"
    streak_start = _parse_dt(rows[0]["timestamp"])

    for row in rows[1:]:
        mode = "REST" if row["is_distracted"] else "FOCUS"
        if mode != current_mode:
            break
        streak_start = _parse_dt(row["timestamp"])

    return (current_mode, streak_start)


# ---------------------------------------------------------------------------
# Capture operations
# ---------------------------------------------------------------------------


def save_capture(
    session_id: int,
    focus_score: int,
    is_distracted: bool,
    activity_description: str,
    feedback_message: str,
    suggestions: list[str],
    activity_label: Optional[str] = None,
    distraction_category: Optional[str] = None,
    suggested_next_interval: Optional[int] = None,
    break_quality_score: Optional[int] = None,
    posture_correction: Optional[str] = None,
    keystroke_count: Optional[int] = None,
    mouse_distance_px: Optional[float] = None,
    click_count: Optional[int] = None,
    heart_rate: Optional[int] = None,
    resting_hr: Optional[int] = None,
    hrv: Optional[float] = None,
    steps: Optional[int] = None,
    webcam_image: Optional[bytes] = None,
) -> int:
    """Persist a capture + analysis result and return the record ID."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO captures (
                session_id, timestamp, focus_score, is_distracted,
                activity_description, feedback_message, suggestions,
                webcam_image,
                activity_label, distraction_category,
                suggested_next_interval, break_quality_score, posture_correction,
                keystroke_count, mouse_distance_px, click_count,
                heart_rate, resting_hr, hrv, steps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                now,
                focus_score,
                int(is_distracted),
                activity_description,
                feedback_message,
                json.dumps(suggestions),
                webcam_image,
                activity_label,
                distraction_category,
                suggested_next_interval,
                break_quality_score,
                posture_correction,
                keystroke_count,
                mouse_distance_px,
                click_count,
                heart_rate,
                resting_hr,
                hrv,
                steps,
            ),
        )
        return cur.lastrowid or 0  # type: ignore[return-value]


def _row_to_capture_record(r: sqlite3.Row, load_images: bool = False) -> CaptureRecord:
    return CaptureRecord(
        id=r["id"],
        session_id=r["session_id"],
        timestamp=_parse_dt(r["timestamp"]),
        focus_score=r["focus_score"],
        is_distracted=bool(r["is_distracted"]),
        activity_description=r["activity_description"],
        feedback_message=r["feedback_message"],
        suggestions=json.loads(r["suggestions"]),
        activity_label=r["activity_label"],
        distraction_category=r["distraction_category"],
        suggested_next_interval=r["suggested_next_interval"],
        break_quality_score=r["break_quality_score"],
        posture_correction=r["posture_correction"],
        keystroke_count=r["keystroke_count"],
        mouse_distance_px=r["mouse_distance_px"],
        click_count=r["click_count"],
        heart_rate=r["heart_rate"],
        resting_hr=r["resting_hr"],
        hrv=r["hrv"],
        steps=r["steps"],
        webcam_image=r["webcam_image"] if load_images else None,
    )


def get_recent_captures(
    session_id: int,
    limit: int = 10,
    load_images: bool = False,
) -> list[CaptureRecord]:
    """Return the most recent captures for a session, newest first."""
    cols = (
        "id, session_id, timestamp, focus_score, is_distracted, "
        "activity_description, feedback_message, suggestions, "
        "activity_label, distraction_category, suggested_next_interval, "
        "break_quality_score, posture_correction, "
        "keystroke_count, mouse_distance_px, click_count, "
        "heart_rate, resting_hr, hrv, steps"
    )
    if load_images:
        cols += ", webcam_image"

    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT {cols} FROM captures WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

    return [_row_to_capture_record(r, load_images) for r in rows]


def get_all_captures_for_session(session_id: int) -> list[CaptureRecord]:
    """Return all captures for a session in chronological order."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, session_id, timestamp, focus_score, is_distracted, "
            "activity_description, feedback_message, suggestions, "
            "activity_label, distraction_category, suggested_next_interval, "
            "break_quality_score, posture_correction, "
            "keystroke_count, mouse_distance_px, click_count, "
            "heart_rate, resting_hr, hrv, steps "
            "FROM captures WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
    return [_row_to_capture_record(r) for r in rows]


def get_session_stats(session_id: int) -> dict:
    """Return aggregate stats for a session."""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)            AS total_captures,
                AVG(focus_score)    AS avg_focus,
                MAX(focus_score)    AS max_focus,
                MIN(focus_score)    AS min_focus,
                SUM(is_distracted)  AS distracted_count
            FROM captures
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
    if row is None or row["total_captures"] == 0:
        return {
            "total_captures": 0,
            "avg_focus": 0.0,
            "max_focus": 0,
            "min_focus": 0,
            "distracted_count": 0,
            "focused_pct": 0.0,
        }
    total = row["total_captures"]
    distracted = row["distracted_count"] or 0
    return {
        "total_captures": total,
        "avg_focus": round(row["avg_focus"] or 0, 1),
        "max_focus": row["max_focus"] or 0,
        "min_focus": row["min_focus"] or 0,
        "distracted_count": distracted,
        "focused_pct": round((total - distracted) / total * 100, 1),
    }


def get_all_sessions_stats() -> list[dict]:
    """Return per-session aggregate stats for all sessions (for history page)."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.start_time,
                s.end_time,
                s.goal,
                s.summary,
                s.summary_json,
                COUNT(c.id)          AS total_captures,
                AVG(c.focus_score)   AS avg_focus,
                SUM(c.is_distracted) AS distracted_count
            FROM sessions s
            LEFT JOIN captures c ON c.session_id = s.id
            GROUP BY s.id
            ORDER BY s.start_time DESC
            """
        ).fetchall()
    results = []
    for r in rows:
        total = r["total_captures"] or 0
        distracted = r["distracted_count"] or 0
        results.append(
            {
                "id": r["id"],
                "start_time": _parse_dt(r["start_time"]),
                "end_time": _parse_dt(r["end_time"]) if r["end_time"] else None,
                "goal": r["goal"],
                "summary": r["summary"],
                "summary_json": r["summary_json"],
                "total_captures": total,
                "avg_focus": round(r["avg_focus"] or 0, 1),
                "focused_pct": round((total - distracted) / total * 100, 1)
                if total
                else 0.0,
            }
        )
    return results


# ---------------------------------------------------------------------------
# LLM call log operations
# ---------------------------------------------------------------------------


def save_llm_call(
    call_type: str,
    model: str,
    request_text: str,
    response_text: Optional[str] = None,
    token_input: Optional[int] = None,
    token_output: Optional[int] = None,
    latency_ms: Optional[int] = None,
    error: Optional[str] = None,
    session_id: Optional[int] = None,
) -> int:
    """Persist a single LLM call record and return its ID."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO llm_calls (
                session_id, timestamp, call_type, model,
                request_text, response_text,
                token_input, token_output, latency_ms, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                now,
                call_type,
                model,
                request_text,
                response_text,
                token_input,
                token_output,
                latency_ms,
                error,
            ),
        )
        return cur.lastrowid or 0  # type: ignore[return-value]


def _row_to_llm_call_record(r: sqlite3.Row) -> LlmCallRecord:
    return LlmCallRecord(
        id=r["id"],
        session_id=r["session_id"],
        timestamp=_parse_dt(r["timestamp"]),
        call_type=r["call_type"],
        model=r["model"],
        request_text=r["request_text"],
        response_text=r["response_text"],
        token_input=r["token_input"],
        token_output=r["token_output"],
        latency_ms=r["latency_ms"],
        error=r["error"],
    )


def get_llm_calls(
    session_id: Optional[int] = None,
    limit: int = 200,
) -> list[LlmCallRecord]:
    """Return LLM call records, optionally filtered by session, newest first."""
    with _get_conn() as conn:
        if session_id is not None:
            rows = conn.execute(
                "SELECT * FROM llm_calls WHERE session_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM llm_calls ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_llm_call_record(r) for r in rows]


def get_llm_calls_stats(session_id: Optional[int] = None) -> dict:
    """Return aggregate stats for LLM calls, optionally scoped to a session."""
    with _get_conn() as conn:
        if session_id is not None:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)             AS total_calls,
                    SUM(token_input)     AS total_input,
                    SUM(token_output)    AS total_output,
                    AVG(latency_ms)      AS avg_latency_ms,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_count
                FROM llm_calls
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)             AS total_calls,
                    SUM(token_input)     AS total_input,
                    SUM(token_output)    AS total_output,
                    AVG(latency_ms)      AS avg_latency_ms,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_count
                FROM llm_calls
                """
            ).fetchone()
    if row is None:
        return {
            "total_calls": 0,
            "total_input": 0,
            "total_output": 0,
            "avg_latency_ms": 0.0,
            "error_count": 0,
        }
    return {
        "total_calls": row["total_calls"] or 0,
        "total_input": row["total_input"] or 0,
        "total_output": row["total_output"] or 0,
        "avg_latency_ms": round(row["avg_latency_ms"] or 0, 0),
        "error_count": row["error_count"] or 0,
    }
