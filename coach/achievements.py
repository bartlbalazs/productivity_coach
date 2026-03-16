"""Achievement definitions and evaluation logic for the Productivity Coach.

Each Achievement describes one badge the user can earn.  The ``check``
classmethod on the registry evaluates the full stats dict (from
``database.get_achievement_stats``) and returns the list of earned IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Achievement:
    id: str
    icon: str
    title: str
    description: str
    category: str  # focus | consistency | health | spotify | secret
    secret: bool = False
    # progress: optional (current_fn, target) for progress-bar achievements
    progress_target: Optional[int] = None
    progress_key: Optional[str] = None  # key into stats dict
    # condition receives the full stats dict and returns True when unlocked
    condition: Callable[[dict], bool] = field(
        default=lambda _: False, repr=False, compare=False
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


ACHIEVEMENTS: list[Achievement] = [
    # ── Focus & Flow ────────────────────────────────────────────────────────
    Achievement(
        id="deep_work_novice",
        icon="🌱",
        title="Deep Work Novice",
        description="Complete your very first monitored session.",
        category="focus",
        condition=lambda s: s["total_sessions"] >= 1,
    ),
    Achievement(
        id="in_the_zone",
        icon="🎯",
        title="In The Zone",
        description="Maintain avg focus >8/10 for a session longer than 30 minutes.",
        category="focus",
        condition=lambda s: any(
            ss["avg_focus"] > 8 and ss["dur_min"] >= 30 for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="flow_state_god",
        icon="⚡",
        title="Flow State God",
        description="Complete a session with zero distractions (100% focus).",
        category="focus",
        condition=lambda s: any(
            ss["rest_count"] == 0 and ss["total_captures"] >= 4 for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="marathon_runner",
        icon="🏃",
        title="Marathon Runner",
        description="Complete a single session longer than 2 hours.",
        category="focus",
        condition=lambda s: any(ss["dur_min"] >= 120 for ss in s["sessions"]),
    ),
    Achievement(
        id="the_unstoppable",
        icon="💯",
        title="The Unstoppable",
        description="Log 100 total monitored sessions.",
        category="focus",
        progress_target=100,
        progress_key="total_sessions",
        condition=lambda s: s["total_sessions"] >= 100,
    ),
    Achievement(
        id="the_finisher",
        icon="✅",
        title="The Finisher",
        description="Complete all tasks in a session at least once.",
        category="focus",
        condition=lambda s: s["sessions_all_tasks_done"] >= 1,
    ),
    Achievement(
        id="unbreakable",
        icon="🧱",
        title="Unbreakable",
        description="Complete a session >45 min where focus never drops below 7/10.",
        category="focus",
        condition=lambda s: any(
            ss["dur_min"] >= 45 and ss["min_focus"] >= 7 for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="power_hour",
        icon="⚡",
        title="Power Hour",
        description="Log 60 minutes of pure focus time in a single session.",
        category="focus",
        condition=lambda s: any(
            ss["dur_min"] >= 60 and ss["rest_count"] == 0 for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="efficiency_expert",
        icon="📐",
        title="Efficiency Expert",
        description="Average focus >8.5 across 5 consecutive sessions.",
        category="focus",
        condition=lambda s: s["max_consec_high_focus"] >= 5,
    ),
    Achievement(
        id="task_slayer",
        icon="⚔️",
        title="Task Slayer",
        description="Complete 50 total tasks across all sessions.",
        category="focus",
        progress_target=50,
        progress_key="total_tasks_done",
        condition=lambda s: s["total_tasks_done"] >= 50,
    ),
    # ── Consistency & Habits ────────────────────────────────────────────────
    Achievement(
        id="hat_trick",
        icon="🎩",
        title="Hat Trick",
        description="Achieve a 3-day consecutive session streak.",
        category="consistency",
        condition=lambda s: s["longest_streak"] >= 3,
    ),
    Achievement(
        id="weekly_warrior",
        icon="🔥",
        title="Weekly Warrior",
        description="Achieve a 7-day consecutive session streak.",
        category="consistency",
        condition=lambda s: s["longest_streak"] >= 7,
    ),
    Achievement(
        id="iron_will",
        icon="🪙",
        title="Iron Will",
        description="Achieve a 21-day consecutive session streak.",
        category="consistency",
        condition=lambda s: s["longest_streak"] >= 21,
    ),
    Achievement(
        id="habit_master",
        icon="🏆",
        title="Habit Master",
        description="Achieve a 30-day consecutive session streak.",
        category="consistency",
        condition=lambda s: s["longest_streak"] >= 30,
    ),
    Achievement(
        id="early_bird",
        icon="🌅",
        title="Early Bird",
        description="Complete a session that starts before 7:00 AM.",
        category="consistency",
        condition=lambda s: s["has_early_bird"],
    ),
    Achievement(
        id="night_owl",
        icon="🦉",
        title="Night Owl",
        description="Complete a session that starts after 10:00 PM.",
        category="consistency",
        condition=lambda s: s["has_night_owl"],
    ),
    Achievement(
        id="lunch_break_warrior",
        icon="🥗",
        title="Lunch Break Warrior",
        description="Complete a session that starts between 12:00 PM and 1:00 PM.",
        category="consistency",
        condition=lambda s: s["has_lunch_warrior"],
    ),
    Achievement(
        id="weekend_warrior",
        icon="🏖️",
        title="Weekend Warrior",
        description="Complete sessions on both Saturday and Sunday.",
        category="consistency",
        condition=lambda s: s["has_saturday"] and s["has_sunday"],
    ),
    Achievement(
        id="century_club",
        icon="🕰️",
        title="Century Club",
        description="Log 100 total hours of session time.",
        category="consistency",
        progress_target=6000,
        progress_key="total_focus_minutes",
        condition=lambda s: s["total_focus_minutes"] >= 6000,
    ),
    Achievement(
        id="double_header",
        icon="2️⃣",
        title="Double Header",
        description="Complete 2 sessions of 30+ minutes on the same day.",
        category="consistency",
        condition=lambda s: s["double_header_days"] >= 1,
    ),
    # ── Health & Body (Fitbit / Posture) ────────────────────────────────────
    Achievement(
        id="posture_perfect",
        icon="🧍",
        title="Posture Perfect",
        description="Complete a session (>20 min) with zero posture corrections.",
        category="health",
        condition=lambda s: any(
            ss["posture_corrections"] == 0 and ss["dur_min"] >= 20
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="zen_master",
        icon="🧘",
        title="Zen Master",
        description="Achieve high focus (>8/10) with an average heart rate below 65 bpm.",
        category="health",
        condition=lambda s: any(
            ss["focus_avg_hr"] is not None
            and ss["focus_avg_hr"] < 65
            and ss["avg_focus"] > 8
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="the_revenant",
        icon="💀",
        title="The Revenant",
        description="Achieve high focus (>8/10) recorded with a very low HRV (depleted state).",
        category="health",
        condition=lambda s: any(
            ss["min_hrv"] is not None and ss["min_hrv"] < 20 and ss["avg_focus"] > 8
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="active_breather",
        icon="🚶",
        title="Active Breather",
        description="Log significant steps (>200) during REST intervals in a session.",
        category="health",
        condition=lambda s: s["active_breather_sessions"] >= 1,
    ),
    Achievement(
        id="cool_under_pressure",
        icon="🧊",
        title="Cool Under Pressure",
        description="Maintain avg HR < 70 bpm during a long session (>45 min).",
        category="health",
        condition=lambda s: s["cool_under_pressure"],
    ),
    Achievement(
        id="stand_up",
        icon="🦵",
        title="Stand Up",
        description="Log >200 steps during a break interval.",
        category="health",
        condition=lambda s: any(
            ss.get("steps_max") is not None and ss["steps_max"] >= 200
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="deep_sleep_recovery",
        icon="😴",
        title="Deep Sleep Recovery",
        description="Achieve high focus (>8/10) when HRV indicates excellent recovery.",
        category="health",
        condition=lambda s: any(
            ss.get("max_hrv") is not None and ss["max_hrv"] > 60 and ss["avg_focus"] > 8
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="posture_streak",
        icon="🏅",
        title="Posture Streak",
        description="Complete 3 consecutive sessions with zero posture corrections.",
        category="health",
        condition=lambda s: s["posture_free_streak"] >= 3,
    ),
    Achievement(
        id="active_recovery",
        icon="🏃",
        title="Active Recovery",
        description="Log >1000 total steps across all REST intervals in a session.",
        category="health",
        condition=lambda s: s["active_recovery_sessions"] >= 1,
    ),
    # ── Spotify & Audio ─────────────────────────────────────────────────────
    Achievement(
        id="silence_is_golden",
        icon="🔇",
        title="Silence is Golden",
        description="Complete a session >30 min with no music playing.",
        category="spotify",
        condition=lambda s: any(
            ss["spotify_active_count"] == 0
            and ss["total_captures"] >= 4
            and ss["dur_min"] >= 30
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="hard_mode",
        icon="🎸",
        title="Hard Mode",
        description="Achieve >9/10 avg focus while listening to music.",
        category="spotify",
        condition=lambda s: any(
            ss["avg_focus"] > 9 and ss["spotify_active_count"] > 0
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="lo_fi_loyal",
        icon="🎵",
        title="Lo-Fi Loyal",
        description="Complete 10 sessions with music playing throughout.",
        category="spotify",
        progress_target=10,
        progress_key="lo_fi_sessions",
        condition=lambda s: sum(
            1 for ss in s["sessions"] if ss["spotify_active_count"] >= 4
        )
        >= 10,
    ),
    Achievement(
        id="dj_distraction",
        icon="🎧",
        title="DJ Distraction",
        description="Change tracks >20 times in a single session. (Shame badge!)",
        category="spotify",
        condition=lambda s: s["max_track_changes"] >= 20,
    ),
    Achievement(
        id="heavy_rotation",
        icon="🔁",
        title="Heavy Rotation",
        description="Listen to the same artist in 3 consecutive sessions.",
        category="spotify",
        condition=lambda s: s["max_artist_run"] >= 3,
    ),
    Achievement(
        id="diverse_palette",
        icon="🎨",
        title="Diverse Palette",
        description="Listen to >5 different artists in a single session.",
        category="spotify",
        condition=lambda s: s["max_artists_in_session"] >= 5,
    ),
    Achievement(
        id="the_purist",
        icon="💿",
        title="The Purist",
        description="Play a single track on repeat for at least 12 consecutive checks.",
        category="spotify",
        condition=lambda s: s["max_track_repeats"] >= 12,
    ),
    Achievement(
        id="album_listener",
        icon="📀",
        title="Album Listener",
        description="Listen to music from a single artist for a full session >30 min.",
        category="spotify",
        condition=lambda s: s["has_single_artist_long_session"],
    ),
    Achievement(
        id="genre_hopper",
        icon="🎼",
        title="Genre Hopper",
        description="Complete 5 sessions with music playing (all time).",
        category="spotify",
        condition=lambda s: sum(
            1 for ss in s["sessions"] if ss["spotify_active_count"] > 0
        )
        >= 5,
    ),
    # ── Secret / Funny ──────────────────────────────────────────────────────
    Achievement(
        id="the_zombie",
        icon="🧟",
        title="The Zombie",
        description="Start a session between 3:00 AM and 5:00 AM. Are you okay?",
        category="secret",
        secret=True,
        condition=lambda s: s["has_zombie"],
    ),
    Achievement(
        id="micro_manager",
        icon="📝",
        title="Micro-Manager",
        description="Write a session goal longer than 50 words.",
        category="secret",
        secret=True,
        condition=lambda s: any(
            len((ss["goal"] or "").split()) > 50 for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="blink_of_an_eye",
        icon="👁️",
        title="Blink of an Eye",
        description="Finish a session in under 2 minutes.",
        category="secret",
        secret=True,
        condition=lambda s: any(0 < ss["dur_min"] < 2 for ss in s["sessions"]),
    ),
    Achievement(
        id="ghost_in_the_machine",
        icon="👻",
        title="Ghost in the Machine",
        description="Complete a focus session with zero keyboard and mouse input logged.",
        category="secret",
        secret=True,
        condition=lambda s: any(
            ss["total_captures"] >= 4
            and ss["rest_count"] == 0
            and ss["total_keystrokes"] == 0
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="keyboard_warrior",
        icon="⌨️",
        title="Keyboard Warrior",
        description="Type >5,000 keystrokes in a single session.",
        category="secret",
        secret=True,
        condition=lambda s: any(ss["total_keystrokes"] >= 5000 for ss in s["sessions"]),
    ),
    Achievement(
        id="the_imposter",
        icon="🤡",
        title="The Imposter",
        description="Set a session goal of just 'Work', 'Stuff', or similarly vague.",
        category="secret",
        secret=True,
        condition=lambda s: any(
            (ss["goal"] or "").strip().lower()
            in {"work", "stuff", "things", "misc", "idk", "..."}
            for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="time_traveler",
        icon="🕰️",
        title="Time Traveler",
        description="Complete a session that spans across midnight.",
        category="secret",
        secret=True,
        condition=lambda s: s["spans_midnight"],
    ),
    Achievement(
        id="ghost_protocol",
        icon="🫥",
        title="Ghost Protocol",
        description="Complete a focus session with zero keyboard/mouse input for >10 minutes.",
        category="secret",
        secret=True,
        condition=lambda s: s["ghost_protocol"],
    ),
    Achievement(
        id="404_focus_not_found",
        icon="❌",
        title="404: Focus Not Found",
        description="Avg focus score below 3 for a session longer than 15 minutes.",
        category="secret",
        secret=True,
        condition=lambda s: any(
            ss["avg_focus"] < 3 and ss["dur_min"] >= 15 for ss in s["sessions"]
        ),
    ),
    Achievement(
        id="rage_quit",
        icon="💢",
        title="Rage Quit",
        description="Complete 20 sessions total. You clearly can't stop.",
        category="secret",
        secret=True,
        condition=lambda s: s["total_sessions"] >= 20,
    ),
]

# Map id → Achievement for fast lookup
ACHIEVEMENT_MAP: dict[str, Achievement] = {a.id: a for a in ACHIEVEMENTS}

# Category display config: (label, icon, color)
CATEGORY_META: dict[str, tuple[str, str, str]] = {
    "focus": ("Focus & Flow", "🧠", "#4a90d9"),
    "consistency": ("Consistency & Habits", "📅", "#2ecc71"),
    "health": ("Health & Body", "💪", "#1abc9c"),
    "spotify": ("Spotify & Audio", "🎵", "#1DB954"),
    "secret": ("Secret Achievements", "🤫", "#9b59b6"),
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def evaluate_achievements(stats: dict) -> set[str]:
    """Return the set of achievement IDs that are currently unlocked.

    Args:
        stats: Dict returned by ``database.get_achievement_stats()``.

    Returns:
        Set of unlocked achievement IDs.
    """
    unlocked: set[str] = set()
    for ach in ACHIEVEMENTS:
        try:
            if ach.condition(stats):
                unlocked.add(ach.id)
        except Exception:
            # Never crash the UI because of a broken condition
            pass
    return unlocked


def get_progress(ach: Achievement, stats: dict) -> tuple[int, int] | None:
    """Return (current, target) for progress-bar achievements, or None."""
    if ach.progress_target is None:
        return None
    if ach.progress_key:
        current = int(stats.get(ach.progress_key) or 0)
    else:
        return None
    return current, ach.progress_target
