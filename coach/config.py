"""Configuration for the Productivity Coach application."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # GCP project for Vertex AI (resolved from ADC if not set)
    gcp_project: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    )

    # Vertex AI endpoint location — use "global" for the global endpoint
    vertex_location: str = field(
        default_factory=lambda: os.environ.get("VERTEX_LOCATION", "global")
    )

    # Gemini model to use via Vertex AI.
    model: str = field(
        default_factory=lambda: os.environ.get("COACH_MODEL", "gemini-3.1-pro-preview")
    )

    # Capture interval range in seconds (randomised between min and max)
    capture_interval_min: int = field(
        default_factory=lambda: int(os.environ.get("CAPTURE_INTERVAL_MIN", "70"))
    )
    capture_interval_max: int = field(
        default_factory=lambda: int(os.environ.get("CAPTURE_INTERVAL_MAX", "120"))
    )

    # How many recent analyses to send as context to the LLM
    history_context_size: int = field(
        default_factory=lambda: int(os.environ.get("HISTORY_CONTEXT_SIZE", "15"))
    )

    # Local SQLite database path
    db_path: str = field(
        default_factory=lambda: os.environ.get(
            "COACH_DB_PATH",
            os.path.join(os.path.expanduser("~"), ".coach", "coach.db"),
        )
    )

    # Webcam device index (0 = default camera)
    webcam_index: int = field(
        default_factory=lambda: int(os.environ.get("WEBCAM_INDEX", "0"))
    )

    # JPEG quality for webcam captures (0-100)
    image_quality: int = field(
        default_factory=lambda: int(os.environ.get("IMAGE_QUALITY", "60"))
    )

    # Maximum image dimension (pixels) for webcam frames.
    webcam_max_dimension: int = field(
        default_factory=lambda: int(os.environ.get("WEBCAM_MAX_DIMENSION", "320"))
    )

    # Retry settings for analyse_node (exponential backoff with full jitter)
    analyse_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("ANALYSE_MAX_RETRIES", "3"))
    )
    analyse_retry_base_delay: float = field(
        default_factory=lambda: float(os.environ.get("ANALYSE_RETRY_BASE_DELAY", "2.0"))
    )

    # Spotify OAuth credentials (optional — Spotify integration is disabled when empty)
    spotify_client_id: str = field(
        default_factory=lambda: os.environ.get("SPOTIFY_CLIENT_ID", "")
    )
    spotify_client_secret: str = field(
        default_factory=lambda: os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    )
    # Redirect URI — must match what is registered in the Spotify Developer Dashboard.
    # spotipy will start a temporary local HTTP server on this port during the OAuth flow.
    spotify_redirect_uri: str = field(
        default_factory=lambda: os.environ.get(
            "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8765"
        )
    )
    # Path to the cached Spotify OAuth token file
    spotify_token_path: str = field(
        default_factory=lambda: os.path.expanduser(
            os.environ.get("SPOTIFY_TOKEN_PATH", "~/.coach/spotify_token.json")
        )
    )

    # Fitbit OAuth credentials (optional — Fitbit integration is disabled when empty)
    fitbit_client_id: str = field(
        default_factory=lambda: os.environ.get("FITBIT_CLIENT_ID", "")
    )
    fitbit_client_secret: str = field(
        default_factory=lambda: os.environ.get("FITBIT_CLIENT_SECRET", "")
    )
    # Redirect URI — must match what is registered in the Fitbit Developer Console.
    # A temporary local HTTP server will listen on this port during the OAuth flow.
    fitbit_redirect_uri: str = field(
        default_factory=lambda: os.environ.get(
            "FITBIT_REDIRECT_URI", "http://127.0.0.1:8766"
        )
    )
    # Path to the cached Fitbit OAuth token file
    fitbit_token_path: str = field(
        default_factory=lambda: os.path.expanduser(
            os.environ.get("FITBIT_TOKEN_PATH", "~/.coach/fitbit_token.json")
        )
    )

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: list[str] = []

        if self.capture_interval_min >= self.capture_interval_max:
            errors.append(
                "CAPTURE_INTERVAL_MIN must be less than CAPTURE_INTERVAL_MAX."
            )
        if not (0 <= self.image_quality <= 100):
            errors.append("IMAGE_QUALITY must be between 0 and 100.")
        return errors


# Singleton — import and use `config` everywhere
config = Config()
