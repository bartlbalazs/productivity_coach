"""Spotify integration — OAuth2, currently-playing track, local callback server."""

from __future__ import annotations

import html as _html
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Scope required for reading currently-playing track
_SCOPE = "user-read-currently-playing user-read-playback-state"

# How long the local callback server waits before timing out (seconds)
_AUTH_TIMEOUT = 60

# ---------------------------------------------------------------------------
# In-process auth-state cache
# ---------------------------------------------------------------------------
# is_authenticated() is called on every Streamlit render (UI thread).
# Spotipy's get_cached_token() will transparently refresh an expired token,
# which is a blocking network call (~1-5 s) that stalls the entire page render.
# We avoid that by caching the result for _AUTH_CACHE_TTL seconds.  The token
# is refreshed naturally in the background when get_now_playing() actually
# constructs a spotipy.Spotify client (called from the capture thread).
_AUTH_CACHE_TTL = 30  # seconds
_auth_cache_value: bool = False
_auth_cache_ts: float = 0.0
_auth_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SpotifyTrack:
    """Currently-playing Spotify track."""

    track_name: str
    artist_name: str
    is_playing: bool  # True = actively playing, False = paused


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _get_auth_manager():
    """Build a SpotifyOAuth manager backed by ~/.coach/spotify_token.json."""
    from spotipy.cache_handler import CacheFileHandler
    from spotipy.oauth2 import SpotifyOAuth

    from coach.config import config

    return SpotifyOAuth(
        client_id=config.spotify_client_id,
        client_secret=config.spotify_client_secret,
        redirect_uri=config.spotify_redirect_uri,
        scope=_SCOPE,
        cache_handler=CacheFileHandler(cache_path=config.spotify_token_path),
        open_browser=False,
    )


def is_configured() -> bool:
    """True if both SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are set."""
    from coach.config import config

    return bool(config.spotify_client_id and config.spotify_client_secret)


def _invalidate_auth_cache() -> None:
    """Force the next is_authenticated() call to re-check the token file."""
    global _auth_cache_ts
    with _auth_cache_lock:
        _auth_cache_ts = 0.0


def is_authenticated() -> bool:
    """True if a valid (or auto-refreshable) cached token exists.

    Uses an in-process cache (TTL=_AUTH_CACHE_TTL) so that the underlying
    get_cached_token() — which may trigger a blocking network refresh — is
    only called once every 30 s instead of on every Streamlit render.
    """
    global _auth_cache_value, _auth_cache_ts
    if not is_configured():
        return False
    now = time.monotonic()
    with _auth_cache_lock:
        if now - _auth_cache_ts < _AUTH_CACHE_TTL:
            return _auth_cache_value
    # Cache miss — do the real (potentially blocking) check outside the lock
    # so other threads are not held while the network call is in flight.
    try:
        token = _get_auth_manager().get_cached_token()
        result = token is not None
    except Exception:
        result = False
    with _auth_cache_lock:
        _auth_cache_value = result
        _auth_cache_ts = time.monotonic()
    return result


def get_auth_url(state: str = "") -> str:
    """Return the Spotify authorization URL the user must open in their browser."""
    mgr = _get_auth_manager()
    url = mgr.get_authorize_url()
    if state:
        # Append state parameter for CSRF protection
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}state={state}"
    return url


def disconnect() -> None:
    """Delete the cached token file, effectively logging out."""
    import os

    from coach.config import config

    try:
        os.remove(config.spotify_token_path)
        logger.info("Spotify token removed from %s", config.spotify_token_path)
    except FileNotFoundError:
        pass
    _invalidate_auth_cache()


# ---------------------------------------------------------------------------
# Local OAuth callback server
# ---------------------------------------------------------------------------


class AuthServer:
    """
    Temporary local HTTP server that captures the Spotify OAuth callback.

    Lifecycle
    ---------
    1. `start_auth_server()` creates and starts this object.
    2. The user opens `get_auth_url()` in their browser.
    3. After Spotify redirects to http://127.0.0.1:<port>?code=..., the server
       exchanges the code for tokens, saves them, and shuts down.
    4. `.completed` is True on success; `.timed_out` is True if no callback
       arrived within _AUTH_TIMEOUT seconds; `.error` holds any error message.
    """

    def __init__(self) -> None:
        self.completed: bool = False
        self.timed_out: bool = False
        self.error: Optional[str] = None
        self._server: Optional[HTTPServer] = None
        self._started_at: float = time.monotonic()
        self._expected_state: str = ""

    def start(self) -> None:
        """Start the HTTP server and timeout watchdog in daemon threads."""
        from coach.config import config

        # Generate CSRF state token
        self._expected_state = secrets.token_urlsafe(16)

        # Parse port from redirect_uri (default 8765)
        parsed = urlparse(config.spotify_redirect_uri)
        port = parsed.port or 8765

        auth_server_ref = self  # closure for the handler

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                qs = parse_qs(urlparse(self.path).query)

                def _shutdown_server() -> None:
                    if auth_server_ref._server is not None:
                        auth_server_ref._server.shutdown()
                        auth_server_ref._server.server_close()

                # Validate state to prevent CSRF
                received_state = qs.get("state", [""])[0]
                if received_state != auth_server_ref._expected_state:
                    auth_server_ref.error = (
                        "OAuth state mismatch — possible CSRF attempt."
                    )
                    self._respond("Authorization failed: state mismatch.")
                    threading.Thread(target=_shutdown_server, daemon=True).start()
                    return

                code_list = qs.get("code")
                if not code_list:
                    # Redirect without code (e.g. error=access_denied)
                    error = qs.get("error", ["unknown"])[0]
                    auth_server_ref.error = f"Spotify denied access: {error}"
                    self._respond("Authorization failed.")
                    threading.Thread(target=_shutdown_server, daemon=True).start()
                    return

                code = code_list[0]
                try:
                    mgr = _get_auth_manager()
                    mgr.get_access_token(code, as_dict=False, check_cache=False)
                    auth_server_ref.completed = True
                    _invalidate_auth_cache()
                    logger.info("Spotify OAuth completed successfully.")
                    self._respond(
                        "Spotify connected! You can close this tab and return to the app."
                    )
                except Exception as exc:
                    auth_server_ref.error = str(exc)
                    logger.warning("Spotify token exchange failed: %s", exc)
                    self._respond(f"Authorization error: {_html.escape(str(exc))}")
                finally:
                    threading.Thread(target=_shutdown_server, daemon=True).start()

            def _respond(self, message: str) -> None:
                body = (
                    f"<html><body style='font-family:sans-serif;padding:2rem'>"
                    f"<h2>{message}</h2></body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args) -> None:  # type: ignore[override]  # noqa: A002
                pass  # suppress default HTTP access log noise

        try:
            self._server = HTTPServer(("127.0.0.1", port), _Handler)
        except OSError as exc:
            self.error = f"Could not start callback server on port {port}: {exc}"
            logger.error(self.error)
            return

        server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="spotify-oauth-server"
        )
        server_thread.start()

        # Timeout watchdog
        timeout_thread = threading.Thread(
            target=self._watchdog, daemon=True, name="spotify-oauth-watchdog"
        )
        timeout_thread.start()
        logger.info("Spotify OAuth callback server listening on port %d", port)

    def _watchdog(self) -> None:
        """Shut down the server after _AUTH_TIMEOUT seconds if not yet completed."""
        time.sleep(_AUTH_TIMEOUT)
        if not self.completed and self._server is not None:
            self.timed_out = True
            logger.info("Spotify OAuth callback timed out after %ds.", _AUTH_TIMEOUT)
            self._server.shutdown()
            self._server.server_close()

    @property
    def is_pending(self) -> bool:
        """True while the server is still waiting for the callback."""
        return not self.completed and not self.timed_out and self.error is None


def start_auth_server() -> AuthServer:
    """Create, start, and return an AuthServer instance."""
    server = AuthServer()
    server.start()
    return server


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def get_now_playing() -> Optional[SpotifyTrack]:
    """
    Return the currently-playing track, or None.

    Returns None silently when:
    - Spotify is not configured / authenticated
    - Nothing is playing
    - Any API or network error occurs
    """
    if not is_authenticated():
        return None
    try:
        import spotipy

        sp = spotipy.Spotify(auth_manager=_get_auth_manager())
        data = sp.current_playback()
        if data is None or data.get("item") is None:
            return None
        item = data["item"]
        track_name: str = item.get("name", "Unknown")
        artists: str = ", ".join(a["name"] for a in item.get("artists", []))
        is_playing: bool = bool(data.get("is_playing", False))
        return SpotifyTrack(
            track_name=track_name,
            artist_name=artists,
            is_playing=is_playing,
        )
    except Exception as exc:
        logger.debug("Spotify get_now_playing error (silently ignored): %s", exc)
        return None
