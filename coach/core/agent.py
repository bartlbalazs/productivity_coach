"""LangGraph agent that drives a full capture → analyse → persist cycle."""

from __future__ import annotations

import base64
import logging
import random
import re
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from coach.core.capture import CaptureResult, capture_all
from coach.config import config
from coach.database import (
    CaptureRecord,
    SessionLogEntry,
    get_all_captures_for_session,
    get_mode_streak_start,
    get_recent_captures,
    get_session_log,
    save_capture,
    save_llm_call,
)
from coach.integrations.input_monitor import InputMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0.3, model_name: str | None = None) -> ChatVertexAI:
    """Return a ChatVertexAI instance using ADC, targeting the global Vertex endpoint."""
    return ChatVertexAI(
        model_name=model_name or config.model,
        project=config.gcp_project or None,
        location=config.vertex_location,
        temperature=temperature,
    )


def _serialize_messages(messages: list) -> str:
    """Serialise a LangChain message list to a readable string for DB storage.

    Image data (base64 or raw bytes) is replaced with a size placeholder so the
    DB stays lean.  Text content is preserved verbatim.
    """
    parts = []
    for msg in messages:
        role = getattr(msg, "__class__", type(msg)).__name__.replace("Message", "")
        content = msg.content
        if isinstance(content, str):
            parts.append(f"[{role}]\n{content}")
        elif isinstance(content, list):
            # Multimodal content: list of dicts with 'type' key
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        # Replace base64 data with size info
                        m = re.match(r"data:(image/\w+);base64,(.+)", url)
                        if m:
                            mime, b64 = m.group(1), m.group(2)
                            kb = round(len(b64) * 3 / 4 / 1024)
                            text_parts.append(f"<{mime} image: ~{kb} KB>")
                        else:
                            text_parts.append("<image>")
                    else:
                        text_parts.append(str(item))
                else:
                    text_parts.append(str(item))
            parts.append(f"[{role}]\n" + "\n".join(text_parts))
        else:
            parts.append(f"[{role}]\n{content!s}")
    return "\n\n".join(parts)


def _extract_usage(response) -> tuple[Optional[int], Optional[int]]:
    """Extract (input_tokens, output_tokens) from a LangChain response if available."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        # Try response_metadata dict
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("usage_metadata") or meta.get("token_usage")
    if usage is None:
        return None, None
    if hasattr(usage, "input_tokens"):
        return usage.input_tokens, usage.output_tokens
    if isinstance(usage, dict):
        inp = usage.get("input_tokens") or usage.get("prompt_tokens")
        out = usage.get("output_tokens") or usage.get("completion_tokens")
        return inp, out
    return None, None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def health_check() -> None:
    """
    Verify that the Gemini API is reachable and the configured model responds
    before the first cycle.

    Sends a minimal text-only prompt to the model.  Raises an exception with a
    human-readable message if anything is wrong (invalid key, quota exceeded, etc.).
    """
    llm = _get_llm(temperature=0.0)
    logger.info("Running Vertex AI health check (model=%s)...", config.model)
    prompt_msgs = [HumanMessage(content="Reply with the single word: OK")]
    request_text = _serialize_messages(prompt_msgs)
    t0 = time.perf_counter()
    try:
        response = llm.invoke(prompt_msgs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        response_text = str(response.content)
        tok_in, tok_out = _extract_usage(response)
        logger.info("Health check passed. Response: %s", response_text)
        save_llm_call(
            call_type="health_check",
            model=config.model,
            request_text=request_text,
            response_text=response_text,
            token_input=tok_in,
            token_output=tok_out,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        save_llm_call(
            call_type="health_check",
            model=config.model,
            request_text=request_text,
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class UserMode(str, Enum):
    FOCUS = "focus"
    REST = "rest"


class AnalysisResult(BaseModel):
    """Structured analysis output from the LLM."""

    mode: UserMode = Field(
        description=(
            "The REQUIRED mode right now. 'focus' for deep work, 'rest' for recovery. "
            "Switch to 'rest' ONLY when a sprint is complete (~25m) or fatigue is evident. "
            "NEVER reward distraction with rest."
        )
    )
    focus_score: int = Field(
        description="Focus score (1-10). 10 = Flow State. 1 = Chaos. Be strict.",
        ge=1,
        le=10,
    )
    activity_description: str = Field(
        description=(
            "1-3 sentences. Brutally honest observation. "
            "If focused: validate the specific task. "
            "If distracted: call out the exact time-wasting behavior."
        )
    )
    activity_label: str = Field(
        description="Context-free label. Max 4 words. E.g., 'Deep Coding', 'Doomscrolling'."
    )
    instruction: str = Field(
        description=(
            "The single most critical command. Max 15 words. Imperative. "
            "Address the user directly as 'you'. No softening. "
            "Focus: 'Close Discord and fix the API.' Rest: 'Stand up and walk away.'"
        )
    )
    suggestions: list[str] = Field(
        description=(
            "Exactly 3 actionable micro-commands. Max 6 words each. "
            "Must relate to the current state (e.g., 'Phone face down', 'Shoulders back', 'Switch to Lo-Fi')."
        ),
        min_length=3,
        max_length=3,
    )
    distraction_category: Optional[str] = Field(
        default=None,
        description=(
            "Only populate when focus_score <= 6. "
            "Pick the single best-matching category: "
            "social_media | messaging | browsing | video | phone | conversation | "
            "fatigue | environment | unknown. "
            "Null when focused."
        ),
    )
    suggested_next_interval: Optional[int] = Field(
        default=None,
        description=(
            "Seconds until the next check-in. Range 60–600. "
            "During FOCUS — SHORT (60–120s): mode just changed or focus is unstable; "
            "LONG (300–600s): deeply focused and steady mid-sprint. "
            "During REST — set this to the FULL break duration: "
            "300s (5 min) for a regular short break, "
            "900s (15 min) for the long break after every 4th sprint. "
            "Null = use the scheduler default."
        ),
        ge=60,
        le=600,
    )
    posture_correction: Optional[str] = Field(
        default=None,
        description=(
            "Only populate if posture is poor (webcam available). "
            "PRIORITY: Detect 'tech neck' (forward head) or 'turtle shoulders' (shrugged). These cause pain. "
            "Command a specific orthopedic fix: 'Tuck chin', 'Roll shoulders back', 'Spine vertical'. "
            "Max 40 words. Address as 'you'."
        ),
    )

    @field_validator("posture_correction", "distraction_category", mode="before")
    @classmethod
    def _normalise_null_string(cls, v: object) -> object:
        """Convert LLM-emitted string 'null'/'none'/whitespace-only to Python None."""
        if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
            return None
        return v


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class CycleState(TypedDict):
    # Inputs
    session_id: int
    session_goal: Optional[str]
    input_monitor: Optional[InputMonitor]  # passed in from scheduler
    tasks_empty: bool  # True when no active tasks are defined for this session

    # Populated by the capture node
    capture: Optional[CaptureResult]
    history: list[CaptureRecord]

    # Populated by the analyse node
    messages: Annotated[Sequence[BaseMessage], add_messages]
    result: Optional[AnalysisResult]

    # Populated by the persist node
    record_id: Optional[int]
    timestamp: Optional[datetime]
    completed_sprints: int

    # Populated by the capture node — true streak start from full session history
    true_streak_start: Optional[tuple[str, datetime]]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an elite productivity coach AI. Your sole purpose is to extract maximum deep work output from the person you are coaching.
Address the user directly as "you". Never "the user".
Your tone is concise, commanding, and psychologically precise. Use urgency and identity-based motivation ("You are a professional").

--- POMODORO CYCLE RULES (Strict) ---
1.  **Focus Sprint:** Minimum 20 min. Ideal 25-30 min. Pure, single-task execution. NEVER recommend REST before 20 minutes unless physical pain/emergency.
2.  **Short Rest:** 5 min after each sprint. Mandatory. Set suggested_next_interval=300.
3.  **Long Rest:** 15 min after every 4th sprint. Non-negotiable for cognitive recovery. Set suggested_next_interval=900.
*   **Switch to REST when:** Sprint is done (25m+) AND focus is declining, OR hard cap (40m) reached.
*   **Switch to FOCUS when:** Rest time is up.

--- SENSORY INTELLIGENCE ---
**CRITICAL:** IF DATA IS MISSING (e.g., "Webcam unavailable", "No Spotify data"), DO NOT HALLUCINATE IT. Ignore that sense entirely.
*   **Digital Environment:**
    *   **Active Window:** Is this the stated goal? If not, block it.
    *   **Open Windows:** These represent cognitive load. If unrelated apps (chat, social, unused tools) are open, only flag them if focus is unstable (≤6). If the user is locked in (>6), IGNORE background windows.
*   **Visual (Webcam) - ERGONOMICS FIRST:**
    *   **CRITICAL:** Detect "Tech Neck" (forward head posture) and "Turtle Shoulders" (shrugged). These cause chronic pain and kill flow.
    *   Command specific orthopedic fixes: "Chin tuck", "Roll shoulders back and down", "Lengthen spine".
    *   Secondary: Look for fatigue (rubbing eyes) or high intensity.
    *   Do not nag unless posture is actively harmful.
    *   *If webcam is unavailable, skip all posture checks.*
*   **Bio-Signals:**
    *   Interpret Heart Rate and Steps relative to the session context.
    *   High HR + Low Focus = Stress/Anxiety -> Suggest a breathing reset.
    *   Sedentary (Low Steps) + Long Session -> Command movement/stretching.
    *   *If no Fitbit data, ignore.*
*   **Audio (Spotify):**
    *   **Lyrical Music (Pop, Hip-Hop) during FOCUS:** High distraction risk. If focus is slipping, command a switch to instrumental.
    *   **Instrumental/Lo-Fi:** generally safe for focus.
    *   **Rest:** Any music is permitted.
    *   *If no track info, ignore.*

--- COACHING DIRECTIVES ---
*   **Protect Flow:** If Focus is 8-10/10, do not interrupt. Validate briefly ("Good pace, keep going") and vanish.
*   **Pain Prevention:** If posture is visibly harmful ("Tech Neck", "Turtle Shoulders"), command a fix immediately, even in focus mode. Physical pain is the enemy of deep work.
*   **Correct Drift:** If Focus drops (≤6), be surgical. Identify the root cause (environment, fatigue, audio, or specific app) and command a fix.
*   **Micro-Target Rule:** When focus drops (≤5), the user is overwhelmed. Shrink the task until execution is inevitable.
    *   Bad: "Work on the report."
    *   Good: "Write the next paragraph for 10 minutes."
*   **Radical Accountability:** Zero tolerance for "fake rest" (scrolling during break). Call it out immediately based on input activity.
*   **You remember previous instructions:** If the user ignored your last command, escalate clarity and urgency.
*   **Empty Task List:** If the context says "Task list: EMPTY", your instruction MUST tell the user to add tasks to their task list before starting work. DO NOT penalize the user or label them as distracted for task planning. Give a high focus score (e.g., 9 or 10) and set mode to 'focus' because defining objectives is essential work. Example: "Add your tasks to the task list before starting."

--- OUTPUT FORMAT ---
*   **Instruction:** Max 15 words. Direct imperative command. MUST align with the determined Mode (Focus vs Rest).
*   **Activity Description:** 1-3 sentences. Brutally honest observation based on OBSERVABLE SIGNALS. Cite specific evidence (Window Title, Posture, Focus Score).
"""


def count_completed_sprints(history: list[CaptureRecord]) -> int:
    """Count completed FOCUS sprints in *history* (oldest-first or newest-first).

    A sprint is counted each time a FOCUS record is immediately followed by a
    REST record when the list is traversed oldest → newest.  The function
    accepts history in either order because it reverses it internally.
    """
    completed = 0
    prev_mode: Optional[str] = None
    for rec in reversed(history):  # oldest first
        rec_mode = rec.mode_label
        if prev_mode == "FOCUS" and rec_mode == "REST":
            completed += 1
        prev_mode = rec_mode
    return completed


def _build_history_summary(
    history: list[CaptureRecord],
    true_streak_start: Optional[tuple[str, datetime]] = None,
) -> str:
    if not history:
        return "No previous observations this session — this is the first check-in."

    lines = ["Recent check-ins (oldest to newest):"]
    # get_recent_captures returns newest-first; reverse to show chronologically.
    for rec in reversed(history):
        dt_str = rec.timestamp.strftime("%H:%M")
        mode_str = rec.mode_label
        lines.append(
            f"  [{dt_str}] {mode_str} — Focus {rec.focus_score}/10 — {rec.activity_description}"
        )

    scores = [r.focus_score for r in history]
    avg = sum(scores) / len(scores)
    lines.append(f"\nTrend: avg focus {avg:.1f}/10 over last {len(history)} check-ins.")

    # Streak duration: prefer the true streak start computed from the full session
    # history (not capped by the context window).  Fall back to estimating from
    # the windowed history if the true start was not provided.
    now = datetime.now(tz=history[0].timestamp.tzinfo)

    if true_streak_start is not None:
        current_mode, streak_start = true_streak_start
    else:
        # Fallback: derive from the windowed history (may undercount long streaks)
        current_mode = history[0].mode_label
        streak_start = history[0].timestamp
        for rec in history[1:]:
            if rec.mode_label != current_mode:
                break
            streak_start = rec.timestamp

    elapsed_min = int((now - streak_start).total_seconds() / 60)
    lines.append(
        f"Current {current_mode} streak: {elapsed_min} min "
        f"(since {streak_start.strftime('%H:%M')}). "
        f"[IMPORTANT: this is the authoritative streak duration — use it to enforce "
        f"Pomodoro rules regardless of how many check-ins are visible above.]"
    )

    # Count completed FOCUS sprints: each uninterrupted FOCUS→REST transition = 1 sprint
    completed_sprints = count_completed_sprints(history)
    if completed_sprints > 0:
        lines.append(
            f"Completed FOCUS sprints this session: {completed_sprints}. "
            + (
                "A long break (15–20 min) is due after the 4th sprint."
                if completed_sprints >= 4 and completed_sprints % 4 == 0
                else ""
            )
        )

    return "\n".join(lines)


def _build_content_parts(
    capture: CaptureResult,
    history: list[CaptureRecord],
    session_goal: Optional[str] = None,
    true_streak_start: Optional[tuple[str, datetime]] = None,
    tasks_empty: bool = False,
) -> list:
    """Build the multimodal HumanMessage content list."""
    parts: list = []

    if session_goal:
        parts.append({"type": "text", "text": f"Session goal: {session_goal}"})

    if tasks_empty:
        parts.append(
            {
                "type": "text",
                "text": "Task list: EMPTY — no active tasks are defined for this session.",
            }
        )

    parts.append(
        {"type": "text", "text": _build_history_summary(history, true_streak_start)}
    )

    if capture.has_webcam:
        b64 = base64.b64encode(capture.webcam_bytes).decode()  # type: ignore[arg-type]
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
        parts.append({"type": "text", "text": "Above: webcam image of you."})
    else:
        parts.append(
            {"type": "text", "text": f"(Webcam unavailable: {capture.webcam_error})"}
        )

    parts.append(
        {
            "type": "text",
            "text": (
                "Please analyse the images and history above, then provide your "
                "structured productivity coaching assessment."
            ),
        }
    )

    # Active window and open windows context — always injected when available.
    # This is the most reliable signal for what the user is actually doing.
    if capture.active_window is not None or capture.open_windows:
        lines = []
        if capture.active_window is not None:
            aw = capture.active_window
            lines.append(f'Active window: {aw.app_name!r} — "{aw.title}"')
        if capture.open_windows:
            other = [w for w in capture.open_windows if not w.is_active]
            if other:
                window_list = ", ".join(f'{w.app_name!r} ("{w.title}")' for w in other)
                lines.append(f"Other open windows: {window_list}")
        parts.append({"type": "text", "text": "\n".join(lines)})

    # In REST mode, include input activity as context so the LLM can reference
    # it in the instruction (e.g. "You're still typing — step away from the keyboard").
    # We infer the current mode from the most recent history entry (is_distracted=True
    # means the last recorded mode was REST).  We skip on the first cycle (no history)
    # and in FOCUS mode where input metrics are not relevant to the LLM.
    last_in_rest = bool(history) and history[0].is_distracted  # history[0] is newest
    if capture.input_snapshot is not None and last_in_rest:
        snap = capture.input_snapshot
        parts.append(
            {
                "type": "text",
                "text": (
                    f"Input activity since last check-in (REST mode context): "
                    f"{snap.keystroke_count} keystrokes, "
                    f"{snap.click_count} mouse clicks, "
                    f"{int(snap.mouse_distance_px)} px mouse movement. "
                    "Use this to assess whether you are genuinely resting "
                    "(low input = good break) or still working (high input = cheating the break)."
                ),
            }
        )

    # Spotify currently-playing track — only injected when actively playing.
    if capture.spotify_track is not None and capture.spotify_track.is_playing:
        t = capture.spotify_track
        parts.append(
            {
                "type": "text",
                "text": (
                    f"--- SPOTIFY ---\n"
                    f'Currently playing: "{t.track_name}" by {t.artist_name}'
                ),
            }
        )

    # Fitbit health snapshot — only injected when data is available.
    if capture.fitbit_data is not None:
        fd = capture.fitbit_data
        lines: list[str] = ["--- FITBIT ---"]
        if fd.heart_rate is not None:
            resting_note = (
                f" (resting HR: {fd.resting_hr} bpm)"
                if fd.resting_hr is not None
                else ""
            )
            lines.append(f"Heart rate: {fd.heart_rate} bpm{resting_note}")
        if fd.hrv is not None:
            lines.append(f"HRV (daily rMSSD): {fd.hrv:.1f} ms")
        if fd.steps is not None:
            lines.append(f"Steps today: {fd.steps:,}")
        if fd.sleep_summary is not None:
            lines.append(f"Last night's sleep: {fd.sleep_summary}")
        if len(lines) > 1:  # only inject if there is actual data beyond the header
            parts.append({"type": "text", "text": "\n".join(lines)})

    return parts


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def capture_node(state: CycleState) -> dict:
    """Capture webcam frame (+ input snapshot) and load history from DB."""
    logger.info("Capturing webcam frame...")
    input_monitor: Optional[InputMonitor] = state.get("input_monitor")  # type: ignore[assignment]
    capture = capture_all(input_monitor=input_monitor)

    history = get_recent_captures(
        state["session_id"],
        limit=config.history_context_size,
    )

    # Fetch the true streak start from the full session history so the LLM
    # gets the accurate elapsed time even when the context window is shorter
    # than the actual streak (e.g. 40-minute unbroken FOCUS session).
    true_streak_start = get_mode_streak_start(state["session_id"])

    logger.info(
        "Capture done. webcam=%s history_entries=%d streak=%s",
        "ok" if capture.has_webcam else "unavailable",
        len(history),
        f"{true_streak_start[0]} since {true_streak_start[1].strftime('%H:%M')}"
        if true_streak_start
        else "none",
    )

    return {
        "capture": capture,
        "history": history,
        "true_streak_start": true_streak_start,
    }


def analyse_node(state: CycleState) -> dict:
    """Send images + history context to the LLM and return a structured AnalysisResult.

    Retries up to config.analyse_max_retries times with exponential backoff + jitter
    to handle transient API errors (model loading, timeout, quota, etc.).
    """
    capture: CaptureResult = state["capture"]  # type: ignore[assignment]
    history: list[CaptureRecord] = state["history"]
    session_id: int = state["session_id"]

    structured_llm = _get_llm(temperature=0.3).with_structured_output(AnalysisResult)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_content_parts(
                capture,
                history,
                state.get("session_goal"),
                state.get("true_streak_start"),
                tasks_empty=state.get("tasks_empty", False),
            )
        ),
    ]
    request_text = _serialize_messages(messages)

    max_retries: int = config.analyse_max_retries
    base_delay: float = config.analyse_retry_base_delay

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            logger.info(
                "Sending capture to Vertex AI (%s), attempt %d/%d...",
                config.model,
                attempt + 1,
                max_retries + 1,
            )
            result: AnalysisResult = structured_llm.invoke(messages)  # type: ignore[assignment]
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "Analysis complete: focus_score=%d, mode=%s",
                result.focus_score,
                result.mode,
            )
            tok_in, tok_out = None, None
            save_llm_call(
                call_type="analyse",
                model=config.model,
                request_text=request_text,
                response_text=result.model_dump_json(),
                token_input=tok_in,
                token_output=tok_out,
                latency_ms=latency_ms,
                session_id=session_id,
            )
            return {"messages": messages, "result": result}
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            last_exc = exc
            if attempt >= max_retries:
                save_llm_call(
                    call_type="analyse",
                    model=config.model,
                    request_text=request_text,
                    error=str(exc),
                    latency_ms=latency_ms,
                    session_id=session_id,
                )
                break
            # Exponential backoff with full jitter: sleep in [0, base * 2^attempt]
            cap = base_delay * (2**attempt)
            sleep_secs = random.uniform(0, cap)
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries + 1,
                exc,
                sleep_secs,
            )
            time.sleep(sleep_secs)

    raise RuntimeError(
        f"analyse_node failed after {max_retries + 1} attempts: {last_exc}"
    ) from last_exc


def persist_node(state: CycleState) -> dict:
    """Save the capture and analysis result to SQLite."""
    capture: CaptureResult = state["capture"]  # type: ignore[assignment]
    result: AnalysisResult = state["result"]  # type: ignore[assignment]
    assert result is not None, (
        "persist_node called before analyse_node produced a result"
    )

    # Compute break quality algorithmically from input metrics (REST mode only).
    # The LLM no longer produces this field — it is derived from keyboard/mouse data.
    snapshot = capture.input_snapshot if capture else None
    break_quality_score: Optional[int] = None
    if result.mode.value == "rest" and snapshot is not None:
        break_quality_score = snapshot.compute_break_quality()

    record_id = save_capture(
        session_id=state["session_id"],
        focus_score=result.focus_score,
        is_distracted=(result.mode.value == "rest"),
        activity_description=result.activity_description,
        activity_label=result.activity_label,
        feedback_message=result.instruction,
        suggestions=result.suggestions,
        distraction_category=result.distraction_category,
        suggested_next_interval=result.suggested_next_interval,
        break_quality_score=break_quality_score,
        posture_correction=result.posture_correction,
        keystroke_count=snapshot.keystroke_count if snapshot else None,
        mouse_distance_px=snapshot.mouse_distance_px if snapshot else None,
        click_count=snapshot.click_count if snapshot else None,
        heart_rate=capture.fitbit_data.heart_rate if capture.fitbit_data else None,
        resting_hr=capture.fitbit_data.resting_hr if capture.fitbit_data else None,
        hrv=capture.fitbit_data.hrv if capture.fitbit_data else None,
        steps=capture.fitbit_data.steps if capture.fitbit_data else None,
        webcam_image=capture.webcam_bytes,
    )

    logger.info(
        "Persisted capture record id=%d for session %d.", record_id, state["session_id"]
    )
    sprints = count_completed_sprints(state.get("history", []))
    return {
        "record_id": record_id,
        "timestamp": datetime.now(timezone.utc),
        "completed_sprints": sprints,
    }


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


class SessionSummary(BaseModel):
    """End-of-session summary generated by the LLM."""

    headline: str = Field(
        description=(
            "One punchy sentence (max 15 words) capturing the session's defining characteristic."
        )
    )
    overall_score: int = Field(
        description="Overall focus score for the session, 1–10.",
        ge=1,
        le=10,
    )
    focus_time_pct: int = Field(
        description="Estimated percentage of session spent in deep focus (0–100).",
        ge=0,
        le=100,
    )
    peak_period: str = Field(
        description="Time range when focus was highest (e.g. '14:05–14:32'). Max 20 chars."
    )
    key_observations: list[str] = Field(
        description=(
            "Exactly 3 short observations about patterns, distractions, or posture trends. "
            "Max 12 words each."
        ),
        min_length=3,
        max_length=3,
    )
    tomorrow_actions: list[str] = Field(
        description=(
            "Exactly 2 specific actions to do before or at the start of the next session. "
            "Max 10 words each. Imperative."
        ),
        min_length=2,
        max_length=2,
    )
    correlation_insights: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 data-driven correlations observed in this session. "
            "Examples: 'Focus dropped after 60 min of unbroken work', "
            "'Browsing distractions peaked mid-afternoon', "
            "'Posture deteriorated as focus declined'. "
            "Max 15 words each. Empty list if not enough data."
        ),
        max_length=3,
    )
    unfinished_items: list[str] = Field(
        default_factory=list,
        description=(
            "Items from the session log that appear unfinished — the user moved on "
            "without logging completion. Infer from the sequence of log entries: if "
            "something was mentioned but never followed up with a resolution or a 'done' "
            "note, list it here. Max 3 items, max 10 words each."
        ),
        max_length=3,
    )


_SUMMARY_PROMPT = """\
You are a high-performance productivity coach. A work session has just ended. \
Analyse the full history below and produce a concise, honest summary.

Be specific. Reference actual times and activities. \
Do not pad. Do not soften. Identify real patterns.

The input contains two sections:
1. SESSION LOG — free-form notes the user wrote during the session to report what \
they were working on. Read these as a running narrative of intent. If an item appears \
in the log but is never followed up with a resolution or "done" note, flag it as \
likely unfinished in the unfinished_items field.
2. CAPTURE TIMELINE — automated check-ins with focus scores and activity descriptions. \
Cross-reference this with the session log to identify gaps between intent and reality."""


def generate_summary(session_id: int) -> SessionSummary:
    """
    Generate an end-of-session summary using the full capture history and session log.
    Called once when the session is stopped.
    """
    history = get_all_captures_for_session(session_id)
    if not history:
        return SessionSummary(
            headline="No data recorded this session.",
            overall_score=5,
            focus_time_pct=0,
            peak_period="N/A",
            key_observations=[
                "No captures were recorded.",
                "Session was too short.",
                "Try again.",
            ],
            tomorrow_actions=["Start session and let it run.", "Stay at your desk."],
        )

    # Build session log section
    log_entries = get_session_log(session_id)
    parts: list[str] = []
    if log_entries:
        log_lines = ["=== SESSION LOG (what you reported during the session) ==="]
        for entry in log_entries:
            ts = entry.timestamp.strftime("%H:%M")
            log_lines.append(f"  [{ts}] {entry.note}")
        parts.append("\n".join(log_lines))
    else:
        parts.append("=== SESSION LOG ===\n  (no notes logged this session)")

    # Build capture timeline section
    timeline_lines = [f"\n=== CAPTURE TIMELINE ({len(history)} check-ins) ==="]
    for rec in history:
        ts = rec.timestamp.strftime("%H:%M")
        mode = rec.mode_label
        line = (
            f"  [{ts}] {mode} — Focus {rec.focus_score}/10 — {rec.activity_description}"
        )
        extras = []
        if rec.activity_label:
            extras.append(f"label={rec.activity_label}")
        if rec.distraction_category:
            extras.append(f"distraction={rec.distraction_category}")
        if rec.break_quality_score is not None:
            extras.append(f"break_quality={rec.break_quality_score}/10")
        if rec.posture_correction:
            extras.append("posture_issue=yes")
        if extras:
            line += f" [{', '.join(extras)}]"
        if rec.feedback_message:
            line += f"\n         Instruction: {rec.feedback_message}"
        timeline_lines.append(line)

    scores = [r.focus_score for r in history]
    avg = sum(scores) / len(scores)
    rest_count = sum(1 for r in history if r.is_distracted)
    timeline_lines.append(
        f"\nOverall: avg focus {avg:.1f}/10, "
        f"{rest_count}/{len(history)} check-ins were in REST mode."
    )
    parts.append("\n".join(timeline_lines))

    timeline_text = "\n\n".join(parts)

    structured_llm = _get_llm(temperature=0.4).with_structured_output(SessionSummary)
    messages = [
        SystemMessage(content=_SUMMARY_PROMPT),
        HumanMessage(content=timeline_text),
    ]
    request_text = _serialize_messages(messages)
    logger.info("Generating session summary for session %d...", session_id)
    t0 = time.perf_counter()
    try:
        summary: SessionSummary = structured_llm.invoke(messages)  # type: ignore[assignment]
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Session summary generated: overall_score=%d", summary.overall_score
        )
        save_llm_call(
            call_type="session_summary",
            model=config.model,
            request_text=request_text,
            response_text=summary.model_dump_json(),
            latency_ms=latency_ms,
            session_id=session_id,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        save_llm_call(
            call_type="session_summary",
            model=config.model,
            request_text=request_text,
            error=str(exc),
            latency_ms=latency_ms,
            session_id=session_id,
        )
        raise
    return summary


# ---------------------------------------------------------------------------
# Task extraction
# ---------------------------------------------------------------------------


class _TaskList(BaseModel):
    """Structured output schema for extract_tasks."""

    tasks: list[str] = Field(
        description=(
            "A flat list of discrete tasks inferred from the session log entries. "
            "Each item should be a short, actionable description (max 10 words). "
            "Merge duplicate or related entries into a single task. "
            "Do NOT include tasks that appear clearly completed in the log."
        )
    )


_TASK_EXTRACTION_PROMPT = """\
You are a task extraction assistant. Given a sequence of timestamped session log \
notes written by the user during a work session, extract a clean list of discrete \
tasks they are working on or intend to work on.

Rules:
- Merge closely related or duplicate entries into a single task.
- Split compound notes into separate tasks when clearly distinct.
- Infer intent from phrasing — "looking into X" and "debugging X" are the same task.
- Keep each task short and actionable (max 10 words).
- Do not include tasks that the notes explicitly mark as finished.
- IMPORTANT: If a list of already-completed tasks is provided, do NOT re-create any \
of them, even if the log still mentions them.
- Return an empty list if there is nothing actionable."""


def extract_tasks(
    log_entries: list[SessionLogEntry],
    done_task_texts: list[str] | None = None,
) -> list[str]:
    """Extract a deduplicated task list from the accumulated session log.

    Args:
        log_entries: All SessionLogEntry objects for the current session.
        done_task_texts: Texts of tasks already marked done — the LLM will not
            re-create these.

    Returns:
        A list of task strings.  Returns [] on failure so the caller can
        continue gracefully without blocking the UI.
    """
    if not log_entries:
        return []

    log_lines = "\n".join(
        f"[{e.timestamp.strftime('%H:%M')}] {e.note}" for e in log_entries
    )

    human_content = log_lines
    if done_task_texts:
        already_done = "\n".join(f"- {t}" for t in done_task_texts)
        human_content += (
            f"\n\nAlready completed (do NOT re-create these):\n{already_done}"
        )

    messages = [
        SystemMessage(content=_TASK_EXTRACTION_PROMPT),
        HumanMessage(content=human_content),
    ]
    request_text = _serialize_messages(messages)

    structured_llm = _get_llm(
        temperature=0.1, model_name="gemini-2.5-flash"
    ).with_structured_output(_TaskList)

    t0 = time.perf_counter()
    try:
        result: _TaskList = structured_llm.invoke(messages)  # type: ignore[assignment]
        latency_ms = int((time.perf_counter() - t0) * 1000)
        save_llm_call(
            call_type="extract_tasks",
            model="gemini-2.5-flash",
            request_text=request_text,
            response_text=result.model_dump_json(),
            latency_ms=latency_ms,
        )
        return result.tasks
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("extract_tasks failed: %s", exc)
        save_llm_call(
            call_type="extract_tasks",
            model="gemini-2.5-flash",
            request_text=request_text,
            error=str(exc),
            latency_ms=latency_ms,
        )
        return []


# ---------------------------------------------------------------------------
# Weekly summary
# ---------------------------------------------------------------------------


class WeeklySummary(BaseModel):
    """Cross-session weekly summary generated by the LLM."""

    headline: str = Field(
        description="One punchy sentence (max 15 words) capturing the week's defining characteristic."
    )
    observations: list[str] = Field(
        description=(
            "Exactly 3 honest observations about focus patterns, common distractions, "
            "or productivity trends across the week. Max 15 words each."
        ),
        min_length=3,
        max_length=3,
    )
    actions: list[str] = Field(
        description=(
            "Exactly 3 specific actions to implement next week to improve. "
            "Max 12 words each. Imperative."
        ),
        min_length=3,
        max_length=3,
    )
    patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 cross-session data-driven patterns. "
            "Examples: 'Focus peaks on mornings, drops after lunch', "
            "'Browsing distractions are most frequent on Wednesdays'. "
            "Max 15 words each. Empty list if insufficient data."
        ),
        max_length=3,
    )


_WEEKLY_SUMMARY_PROMPT = """\
You are a high-performance productivity coach. A week of work sessions has just ended. \
Analyse the multi-session data below and produce an honest, specific, actionable weekly review.

Identify real patterns across days. Reference specific days or trends. Do not pad. Do not soften."""


def generate_weekly_summary(weekly_stats: list[dict]) -> WeeklySummary:
    """
    Generate a cross-session weekly summary from session aggregate stats.

    Args:
        weekly_stats: List of dicts from get_all_sessions_stats() for the past 7 days.

    Returns:
        WeeklySummary with headline, observations, actions, and patterns.
    """
    if not weekly_stats:
        return WeeklySummary(
            headline="No sessions this week.",
            observations=["No data recorded.", "Start monitoring.", "Build the habit."],
            actions=[
                "Start a session today.",
                "Set a session goal.",
                "Run for 30 minutes.",
            ],
        )

    lines = [f"Weekly data: {len(weekly_stats)} sessions in the last 7 days.\n"]
    for s in reversed(weekly_stats):  # oldest first
        day = s["start_time"].strftime("%A %b %d")
        goal_txt = f" | Goal: {s['goal']}" if s.get("goal") else ""
        lines.append(
            f"  {day}: {s['total_captures']} check-ins, "
            f"avg focus {s['avg_focus']}/10, focused {s['focused_pct']}%{goal_txt}"
        )

    total_captures = sum(s["total_captures"] for s in weekly_stats)
    if total_captures > 0:
        weighted_avg = (
            sum(s["avg_focus"] * s["total_captures"] for s in weekly_stats)
            / total_captures
        )
        lines.append(
            f"\nOverall: avg focus {weighted_avg:.1f}/10 across {total_captures} check-ins."
        )

    timeline_text = "\n".join(lines)

    structured_llm = _get_llm(temperature=0.4).with_structured_output(WeeklySummary)
    messages = [
        SystemMessage(content=_WEEKLY_SUMMARY_PROMPT),
        HumanMessage(content=timeline_text),
    ]
    request_text = _serialize_messages(messages)
    logger.info("Generating weekly summary for %d sessions...", len(weekly_stats))
    t0 = time.perf_counter()
    try:
        summary: WeeklySummary = structured_llm.invoke(messages)  # type: ignore[assignment]
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("Weekly summary generated.")
        save_llm_call(
            call_type="weekly_summary",
            model=config.model,
            request_text=request_text,
            response_text=summary.model_dump_json(),
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        save_llm_call(
            call_type="weekly_summary",
            model=config.model,
            request_text=request_text,
            error=str(exc),
            latency_ms=latency_ms,
        )
        raise
    return summary


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    graph = StateGraph(CycleState)

    graph.add_node("capture_images", capture_node)
    graph.add_node("analyse_images", analyse_node)
    graph.add_node("persist_results", persist_node)

    graph.set_entry_point("capture_images")
    graph.add_edge("capture_images", "analyse_images")
    graph.add_edge("analyse_images", "persist_results")
    graph.add_edge("persist_results", END)

    return graph


_compiled_graph = _build_graph().compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_cycle(
    session_id: int,
    session_goal: Optional[str] = None,
    input_monitor: Optional[InputMonitor] = None,
    tasks_empty: bool = False,
) -> CycleState:
    """
    Execute one full capture → analyse → persist cycle via LangGraph.

    Args:
        session_id: Active DB session to load history from and persist into.
        session_goal: Optional goal text passed to the LLM for context-aware coaching.
        input_monitor: Optional InputMonitor whose counters will be snapshotted during
            the capture phase.  Pass None to disable input metric collection.
        tasks_empty: When True, the LLM is told no active tasks are defined and
            should instruct the user to add tasks before starting work.

    Returns:
        The final CycleState, containing the CaptureResult, AnalysisResult,
        persisted record_id, and timestamp.

    Raises:
        RuntimeError: If the graph fails to produce an AnalysisResult.
    """
    initial_state: CycleState = {
        "session_id": session_id,
        "session_goal": session_goal,
        "input_monitor": input_monitor,
        "tasks_empty": tasks_empty,
        "capture": None,
        "history": [],
        "messages": [],
        "result": None,
        "record_id": None,
        "timestamp": None,
        "true_streak_start": None,
        "completed_sprints": 0,
    }
    final_state: CycleState = _compiled_graph.invoke(initial_state)  # type: ignore[assignment]

    if final_state.get("result") is None:
        raise RuntimeError("Cycle graph completed without an AnalysisResult.")

    return final_state
