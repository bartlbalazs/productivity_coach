"""Fitbit integration — OAuth 2.0 PKCE, health data fetching, local callback server."""

from __future__ import annotations

import base64
import hashlib
import html as _html
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests as _requests

logger = logging.getLogger(__name__)

# Scopes required for heart rate, activity, and sleep data
_SCOPE = "heartrate activity sleep"

# Fitbit OAuth 2.0 endpoints
_AUTHORIZE_URL = "https://www.fitbit.com/oauth2/authorize"
_TOKEN_URL = "https://api.fitbit.com/oauth2/token"

# How long the local callback server waits before timing out (seconds)
_AUTH_TIMEOUT = 60

# How many capture cycles between slow-changing metric fetches
# (steps, sleep, HRV — fetched on cycle 0, then every N cycles)
_SLOW_METRIC_INTERVAL = 5

# Module-level cycle counter and cached slow data (reset on new auth).
# Protected by _slow_data_lock for thread safety.
_cycle_count: int = 0
_slow_data_cache: Optional[_SlowMetrics] = None
_slow_data_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class _SlowMetrics:
    """Metrics that change slowly — fetched every _SLOW_METRIC_INTERVAL cycles."""

    resting_hr: Optional[int] = None
    hrv: Optional[float] = None
    steps: Optional[int] = None
    sleep_summary: Optional[str] = None  # e.g. "7h 23m, score 82"


@dataclass
class FitbitData:
    """Health snapshot for one capture cycle."""

    heart_rate: Optional[int] = None  # Most recent intraday HR (bpm)
    resting_hr: Optional[int] = None  # Resting HR for today
    hrv: Optional[float] = None  # Daily HRV (ms)
    steps: Optional[int] = None  # Steps so far today
    sleep_summary: Optional[str] = None  # Last night summary string


# ---------------------------------------------------------------------------
# Token storage helpers
# ---------------------------------------------------------------------------


def _token_path() -> str:
    from coach.config import config

    return config.fitbit_token_path


def _load_token() -> Optional[dict]:
    """Load the cached token dict from disk, or None if not found/invalid."""
    path = _token_path()
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_token(token: dict) -> None:
    """Persist a token dict to disk atomically with owner-only permissions (0o600)."""
    path = _token_path()
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    # Write to a temp file in the same directory, then rename atomically so a
    # crash during write never leaves a truncated token file.
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=".fitbit_token_")
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(token, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _refresh_token_if_needed(token: dict) -> Optional[dict]:
    """
    If the token is expired (or within 60s of expiry), refresh it.
    Returns the updated token dict, or None on failure.
    """
    expires_at = token.get("expires_at", 0)
    if time.time() < expires_at - 60:
        return token  # still valid

    from coach.config import config

    refresh_tok = token.get("refresh_token")
    if not refresh_tok:
        return None

    # Basic auth: base64(client_id:client_secret)
    credentials = base64.b64encode(
        f"{config.fitbit_client_id}:{config.fitbit_client_secret}".encode()
    ).decode()

    try:
        resp = _requests.post(
            _TOKEN_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_tok,
            },
            timeout=10,
        )
        resp.raise_for_status()
        new_token = resp.json()
        new_token["expires_at"] = time.time() + new_token.get("expires_in", 3600)
        _save_token(new_token)
        logger.info("Fitbit token refreshed successfully.")
        return new_token
    except Exception as exc:
        logger.warning("Fitbit token refresh failed: %s", exc)
        return None


def _get_valid_token() -> Optional[dict]:
    """Load and refresh (if needed) the cached token. Returns None if unavailable."""
    token = _load_token()
    if token is None:
        return None
    return _refresh_token_if_needed(token)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def is_configured() -> bool:
    """True if both FITBIT_CLIENT_ID and FITBIT_CLIENT_SECRET are set."""
    from coach.config import config

    return bool(config.fitbit_client_id and config.fitbit_client_secret)


def is_authenticated() -> bool:
    """True if a valid (or auto-refreshable) cached token exists.

    A token is considered valid when it has a refresh_token (so we can always
    renew the access token) or when the access_token is still within its expiry
    window.  A token that has only an expired access_token and no refresh_token
    is treated as unauthenticated.
    """
    if not is_configured():
        return False
    token = _load_token()
    if token is None:
        return False
    # If we have a refresh_token we can always obtain a new access_token
    if token.get("refresh_token"):
        return True
    # No refresh_token — only trust an access_token that is still valid
    expires_at = token.get("expires_at", 0)
    return bool(token.get("access_token")) and time.time() < expires_at - 60


def get_auth_url() -> tuple[str, str, str]:
    """
    Build and return (auth_url, code_verifier, state) for PKCE flow.

    The caller must store code_verifier and state in session state so the
    callback handler can use them to exchange the code for tokens.
    """
    from coach.config import config

    # Generate PKCE code verifier and challenge
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )

    state = secrets.token_urlsafe(16)

    params = {
        "response_type": "code",
        "client_id": config.fitbit_client_id,
        "redirect_uri": config.fitbit_redirect_uri,
        "scope": _SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{_AUTHORIZE_URL}?{urlencode(params)}", code_verifier, state


def disconnect() -> None:
    """Delete the cached token file, effectively logging out."""
    path = _token_path()
    try:
        os.remove(path)
        logger.info("Fitbit token removed from %s", path)
    except FileNotFoundError:
        pass

    # Clear module-level caches so stale data isn't served after reconnect.
    # Hold _slow_data_lock because get_current_health() reads/writes these
    # globals under the same lock from the scheduler thread.
    global _cycle_count, _slow_data_cache
    with _slow_data_lock:
        _cycle_count = 0
        _slow_data_cache = None


# ---------------------------------------------------------------------------
# Local OAuth callback server
# ---------------------------------------------------------------------------


class AuthServer:
    """
    Temporary local HTTP server that captures the Fitbit OAuth callback.

    Lifecycle
    ---------
    1. `start_auth_server()` creates and starts this object.
    2. The user opens the URL returned by `get_auth_url()` in their browser.
    3. After Fitbit redirects to http://127.0.0.1:<port>?code=..., the server
       exchanges the code for tokens via PKCE, saves them, and shuts down.
    4. `.completed` is True on success; `.timed_out` is True if no callback
       arrived within _AUTH_TIMEOUT seconds; `.error` holds any error message.
    """

    def __init__(self) -> None:
        self.completed: bool = False
        self.timed_out: bool = False
        self.error: Optional[str] = None
        self._server: Optional[HTTPServer] = None
        self._started_at: float = time.monotonic()
        # PKCE verifier and state set by start() after generating the auth URL
        self._code_verifier: str = ""
        self._expected_state: str = ""
        # The auth URL is generated once and stored so the UI can link to it
        self.auth_url: str = ""

    def start(self) -> None:
        """Generate PKCE auth URL, start the HTTP server, and start watchdog."""
        from coach.config import config

        auth_url, code_verifier, state = get_auth_url()
        self.auth_url = auth_url
        self._code_verifier = code_verifier
        self._expected_state = state

        parsed = urlparse(config.fitbit_redirect_uri)
        port = parsed.port or 8766

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
                    try:
                        self._respond("Authorization failed: state mismatch.")
                    finally:
                        threading.Thread(target=_shutdown_server, daemon=True).start()
                    return

                code_list = qs.get("code")
                if not code_list:
                    error = qs.get("error", ["unknown"])[0]
                    auth_server_ref.error = f"Fitbit denied access: {error}"
                    try:
                        self._respond("Authorization failed.")
                    finally:
                        threading.Thread(target=_shutdown_server, daemon=True).start()
                    return

                code = code_list[0]
                try:
                    _exchange_code_for_token(code, auth_server_ref._code_verifier)
                    auth_server_ref.completed = True
                    logger.info("Fitbit OAuth completed successfully.")
                    self._respond(
                        "Fitbit connected! You can close this tab and return to the app."
                    )
                except Exception as exc:
                    auth_server_ref.error = str(exc)
                    logger.warning("Fitbit token exchange failed: %s", exc)
                    self._respond(f"Authorization error: {_html.escape(str(exc))}")
                finally:

                    def _shutdown_and_close(srv: HTTPServer) -> None:
                        srv.shutdown()
                        srv.server_close()

                    threading.Thread(
                        target=_shutdown_and_close,
                        args=(auth_server_ref._server,),  # type: ignore[union-attr]
                        daemon=True,
                    ).start()

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

            def log_message(self, *args: object) -> None:  # noqa: ANN002
                pass  # suppress default HTTP access log noise

        try:
            self._server = HTTPServer(("127.0.0.1", port), _Handler)
        except OSError as exc:
            self.error = f"Could not start callback server on port {port}: {exc}"
            logger.error(self.error)
            return

        server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="fitbit-oauth-server"
        )
        server_thread.start()

        # Timeout watchdog
        timeout_thread = threading.Thread(
            target=self._watchdog, daemon=True, name="fitbit-oauth-watchdog"
        )
        timeout_thread.start()
        logger.info("Fitbit OAuth callback server listening on port %d", port)

    def _watchdog(self) -> None:
        """Shut down the server after _AUTH_TIMEOUT seconds if not yet completed."""
        time.sleep(_AUTH_TIMEOUT)
        if not self.completed and self._server is not None:
            self.timed_out = True
            logger.info("Fitbit OAuth callback timed out after %ds.", _AUTH_TIMEOUT)
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


def _exchange_code_for_token(code: str, code_verifier: str) -> None:
    """Exchange an authorization code + PKCE verifier for tokens and save them."""
    from coach.config import config

    credentials = base64.b64encode(
        f"{config.fitbit_client_id}:{config.fitbit_client_secret}".encode()
    ).decode()

    resp = _requests.post(
        _TOKEN_URL,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": config.fitbit_redirect_uri,
            "code_verifier": code_verifier,
        },
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()
    token["expires_at"] = time.time() + token.get("expires_in", 3600)
    _save_token(token)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _api_get(token: dict, path: str) -> Optional[dict]:
    """Make an authenticated GET to the Fitbit API. Returns parsed JSON or None.

    On 401 the token is refreshed and the request is retried once.
    On 429 the Retry-After header is honoured (capped at 60 s).
    """
    for attempt in range(2):
        access_token = token.get("access_token", "")
        try:
            resp = _requests.get(
                f"https://api.fitbit.com{path}",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=8,
            )
            if resp.status_code == 401:
                if attempt == 0:
                    # Token might be stale — force a refresh and retry once.
                    logger.warning(
                        "Fitbit API returned 401 for %s — refreshing token and retrying.",
                        path,
                    )
                    refreshed = _refresh_token_if_needed(
                        {**token, "expires_at": 0}  # force refresh by zeroing expiry
                    )
                    if refreshed is None:
                        logger.warning("Fitbit token refresh failed after 401.")
                        return None
                    token = refreshed
                    continue
                logger.warning(
                    "Fitbit API returned 401 for %s after token refresh.", path
                )
                return None
            if resp.status_code == 429:
                try:
                    retry_after = int(resp.headers.get("Retry-After", "60"))
                except ValueError:
                    # Retry-After may be an HTTP-date string rather than seconds
                    retry_after = 60
                wait = min(retry_after, 60)
                logger.warning(
                    "Fitbit API rate-limited (429) for %s — backing off %ds.",
                    path,
                    wait,
                )
                time.sleep(wait)
                return None  # skip this cycle rather than blocking indefinitely
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("Fitbit API error for %s (silently ignored): %s", path, exc)
            return None
    return None


def _fetch_heart_rate_now(token: dict) -> Optional[int]:
    """Return the most recent intraday heart rate value for today (bpm)."""
    today = date.today().isoformat()
    data = _api_get(token, f"/1/user/-/activities/heart/date/{today}/1d/1min.json")
    if data is None:
        return None
    try:
        intraday = data["activities-heart-intraday"]["dataset"]
        if not intraday:
            return None
        # dataset is chronological; last entry is most recent
        return int(intraday[-1]["value"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _fetch_slow_metrics(token: dict) -> _SlowMetrics:
    """Fetch resting HR, HRV, steps, and sleep summary."""
    result = _SlowMetrics()
    today = date.today().isoformat()

    # Resting heart rate (from daily activity summary)
    hr_data = _api_get(token, f"/1/user/-/activities/heart/date/{today}/1d.json")
    if hr_data:
        try:
            result.resting_hr = hr_data["activities-heart"][0]["value"].get(
                "restingHeartRate"
            )
        except (KeyError, IndexError, TypeError):
            pass

    # HRV (daily)
    hrv_data = _api_get(token, f"/1/user/-/hrv/date/{today}.json")
    if hrv_data:
        try:
            hrv_list = hrv_data.get("hrv", [])
            if hrv_list:
                result.hrv = hrv_list[0].get("value", {}).get("dailyRmssd")
        except (KeyError, IndexError, TypeError):
            pass

    # Steps today
    activity_data = _api_get(token, f"/1/user/-/activities/date/{today}.json")
    if activity_data:
        try:
            result.steps = activity_data["summary"]["steps"]
        except (KeyError, TypeError):
            pass

    # Sleep last night
    sleep_data = _api_get(token, f"/1.2/user/-/sleep/date/{today}.json")
    if sleep_data:
        try:
            summary = sleep_data.get("summary", {})
            total_min = summary.get("totalMinutesAsleep")
            sleep_score = None
            # Sleep score is nested under each log entry's levels.summary
            # but also available via /sleep/date/today/all — use what's available
            for log in sleep_data.get("sleep", []):
                if log.get("isMainSleep"):
                    sleep_score = log.get("efficiency")
                    break
            if total_min is not None:
                hours, mins = divmod(int(total_min), 60)
                parts = [f"{hours}h {mins}m"]
                if sleep_score is not None:
                    parts.append(f"efficiency {sleep_score}%")
                result.sleep_summary = ", ".join(parts)
        except (KeyError, TypeError, ValueError):
            pass

    return result


def get_current_health() -> Optional[FitbitData]:
    """
    Return a FitbitData snapshot for the current capture cycle, or None.

    Returns None silently when:
    - Fitbit is not configured / authenticated
    - Any API or network error occurs

    Rate-limit strategy: heart rate is fetched every cycle; steps, sleep,
    resting HR, and HRV are fetched every _SLOW_METRIC_INTERVAL cycles.

    Thread safety: _cycle_count and _slow_data_cache are protected by
    _slow_data_lock so concurrent callers from different threads cannot
    corrupt state.
    """
    global _cycle_count, _slow_data_cache

    if not is_authenticated():
        return None

    token = _get_valid_token()
    if token is None:
        return None

    try:
        # Always fetch current heart rate (outside the lock — network call)
        heart_rate = _fetch_heart_rate_now(token)

        # Decide whether to refresh slow metrics (lock only for the decision + update)
        with _slow_data_lock:
            need_refresh = (
                _slow_data_cache is None or _cycle_count % _SLOW_METRIC_INTERVAL == 0
            )
            current_count = _cycle_count

        if need_refresh:
            new_slow = _fetch_slow_metrics(token)
            with _slow_data_lock:
                _slow_data_cache = new_slow

        with _slow_data_lock:
            _cycle_count += 1
            slow = _slow_data_cache

        if slow is None:
            return None

        return FitbitData(
            heart_rate=heart_rate,
            resting_hr=slow.resting_hr,
            hrv=slow.hrv,
            steps=slow.steps,
            sleep_summary=slow.sleep_summary,
        )
    except Exception as exc:
        logger.debug("Fitbit get_current_health error (silently ignored): %s", exc)
        return None
