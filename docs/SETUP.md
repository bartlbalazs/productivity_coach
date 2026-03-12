# Setup Guide

Detailed instructions for installing, configuring, and running Productivity Coach, including Spotify and Fitbit integration setup and GCP authentication.

---

## Prerequisites

- **Python 3.12+**
- **[Poetry](https://python-poetry.org/)** — dependency management
- **Google Cloud project** with Vertex AI API enabled
- **[gcloud CLI](https://cloud.google.com/sdk/docs/install)** — for authentication
- **Webcam** — optional but recommended for posture coaching
- **Linux** — desktop notifications use `notify-send`; the rest works on macOS/Windows

---

## Installation

```bash
git clone https://github.com/your-username/coach.git
cd coach
poetry install
```

This installs all dependencies including Streamlit, LangGraph, OpenCV, pygame, pynput, spotipy, and Piper TTS.

---

## GCP authentication (Gemini via Vertex AI)

The app uses [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) to call Gemini through the Vertex AI API. No API key is needed.

### 1. Log in with ADC

```bash
gcloud auth application-default login
```

This opens a browser window. Log in with the Google account that has access to your GCP project. The credentials are cached at `~/.config/gcloud/application_default_credentials.json`.

### 2. Enable the Vertex AI API

```bash
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
```

### 3. Set your project (optional)

The app infers the project from ADC. If you need to override it:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
```

### 4. Model selection

The default model is `gemini-3.1-pro-preview`. Override with:

```bash
export COACH_MODEL=gemini-2.5-flash
```

The `VERTEX_LOCATION` defaults to `global`. Override if your project requires a specific region:

```bash
export VERTEX_LOCATION=us-central1
```

---

## Spotify integration

Spotify integration sends the currently playing track, artist, album, and playback state to the LLM as context each cycle.

### 1. Create a Spotify app

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Click **Create App**
3. Set **Redirect URI** to `http://127.0.0.1:8765`
4. Note the **Client ID** and **Client Secret**

### 2. Set environment variables

```bash
export SPOTIFY_CLIENT_ID=your_client_id
export SPOTIFY_CLIENT_SECRET=your_client_secret
```

Optional overrides (defaults should work):

```bash
export SPOTIFY_REDIRECT_URI=http://127.0.0.1:8765    # default
export SPOTIFY_TOKEN_PATH=~/.coach/spotify_token.json  # default
```

### 3. Connect in the app

1. Start the app (`poetry run streamlit run coach/Coach.py`)
2. In the sidebar, expand the **Spotify** section
3. Click **Connect Spotify**
4. A browser window opens for Spotify authorization
5. After granting access, the sidebar shows the connected state

The token is cached at `~/.coach/spotify_token.json` and refreshes automatically. To disconnect, click **Disconnect** in the sidebar.

---

## Fitbit integration

Fitbit integration reads heart rate, resting HR, HRV, and step count from the Fitbit Web API. Health data is shown on the Session Insights HR overlay chart and included in the LLM system prompt.

### 1. Create a Fitbit app

1. Go to [dev.fitbit.com/apps/new](https://dev.fitbit.com/apps/new)
2. Fill in the form:
   - **Application Name**: any name (e.g. "Productivity Coach")
   - **OAuth 2.0 Application Type**: **Personal** — this auto-grants intraday access for your own data
   - **Redirect URL**: `http://127.0.0.1:8766`
   - **Default Access Type**: Read-Only
3. Note the **OAuth 2.0 Client ID** and **Client Secret**

> **Important**: The redirect URL in the Fitbit Developer Console must exactly match `http://127.0.0.1:8766` (with `http://`, not `https://`). If you registered `https://`, either update it in the console or set `FITBIT_REDIRECT_URI=https://127.0.0.1:8766` as an environment variable.

### 2. Set environment variables

```bash
export FITBIT_CLIENT_ID=your_client_id
export FITBIT_CLIENT_SECRET=your_client_secret
```

Optional overrides:

```bash
export FITBIT_REDIRECT_URI=http://127.0.0.1:8766    # default
export FITBIT_TOKEN_PATH=~/.coach/fitbit_token.json  # default
```

### 3. Connect in the app

1. Start the app
2. In the sidebar, expand the **Fitbit** section
3. Click **Connect Fitbit**
4. A browser window opens for Fitbit authorization (OAuth 2.0 with PKCE)
5. After granting access, the sidebar shows the connected state

The token is cached at `~/.coach/fitbit_token.json` and refreshes automatically.

### Rate limits

Fitbit allows 150 requests per hour. The app uses a rate-limit strategy:

- **Heart rate**: fetched every cycle
- **Steps, HRV, resting HR**: fetched every 5th cycle

---

## Configuration reference

All settings have sensible defaults. Override via environment variables or adjust in the sidebar at runtime (where applicable).

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | *(from ADC)* | GCP project ID |
| `VERTEX_LOCATION` | `global` | Vertex AI endpoint location |
| `COACH_MODEL` | `gemini-3.1-pro-preview` | Gemini model ID |
| `CAPTURE_INTERVAL_MIN` | `70` | Minimum seconds between captures |
| `CAPTURE_INTERVAL_MAX` | `120` | Maximum seconds between captures |
| `HISTORY_CONTEXT_SIZE` | `15` | Recent check-ins sent as LLM context |
| `WEBCAM_INDEX` | `0` | Webcam device index |
| `WEBCAM_MAX_DIMENSION` | `320` | Max webcam image dimension (px) |
| `IMAGE_QUALITY` | `60` | JPEG quality for webcam frames (0-100) |
| `ANALYSE_MAX_RETRIES` | `3` | Retry attempts on Vertex AI API failure |
| `ANALYSE_RETRY_BASE_DELAY` | `2.0` | Base delay (seconds) for exponential backoff |
| `COACH_DB_PATH` | `~/.coach/coach.db` | SQLite database path |
| `SPOTIFY_CLIENT_ID` | *(empty)* | Spotify OAuth client ID |
| `SPOTIFY_CLIENT_SECRET` | *(empty)* | Spotify OAuth client secret |
| `SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:8765` | Spotify OAuth redirect URI |
| `SPOTIFY_TOKEN_PATH` | `~/.coach/spotify_token.json` | Cached Spotify token |
| `FITBIT_CLIENT_ID` | *(empty)* | Fitbit OAuth client ID |
| `FITBIT_CLIENT_SECRET` | *(empty)* | Fitbit OAuth client secret |
| `FITBIT_REDIRECT_URI` | `http://127.0.0.1:8766` | Fitbit OAuth redirect URI |
| `FITBIT_TOKEN_PATH` | `~/.coach/fitbit_token.json` | Cached Fitbit token |

---

## Running the app

```bash
poetry run streamlit run coach/Coach.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

If ADC credentials are not configured, a validation error appears in the UI before you can start a session.

---

## Project structure

```
coach/
├── Coach.py            # Streamlit main page — UI, session lifecycle, event loop
├── config.py           # Environment-variable-driven configuration dataclass
├── database.py         # SQLite schema, migrations, CRUD, stats queries
├── prefs.py            # Sound/TTS preferences (~/.coach/prefs.json)
├── core/
│   ├── agent.py            # LangGraph state graph, LLM prompts, structured output, task extraction
│   ├── capture.py          # Webcam capture, window list, aggregates all data sources
│   ├── scheduler.py        # Background capture loop, Pomodoro enforcement, event queue
│   ├── session_controller.py  # Start/pause/resume/stop session logic
│   └── session_state.py    # Shared in-memory session state
├── integrations/
│   ├── fitbit.py           # Fitbit OAuth 2.0 PKCE, health data (HR, HRV, steps, sleep)
│   ├── input_monitor.py    # Keyboard/mouse activity monitor (pynput)
│   ├── notify.py           # Desktop notifications (notify-send)
│   ├── spotify.py          # Spotify OAuth, currently-playing track
│   └── tts.py              # Piper neural TTS, auto-downloads voice model
├── ui/
│   ├── components.py       # All render functions — sprint timer, task list, coaching card, charts
│   ├── sounds.py           # Synthesised audio cues (pygame + numpy)
│   ├── theme.py            # CSS theme and colour constants
│   └── utils.py            # Shared UI helpers (hide chrome, duration formatting)
└── pages/
    ├── 1_History.py          # Multi-session history, streaks, milestones, weekly AI summary
    ├── 2_LLM_Log.py          # LLM request/response inspector
    ├── 3_Session_Insights.py # Session deep-dive: timeline, charts, HR, input, trends
    └── 4_Settings.py         # Intervals, sound/TTS toggles, model name, Spotify/Fitbit OAuth
```

### Data directory

The app stores all local data at `~/.coach/`:

```
~/.coach/
├── coach.db                # SQLite database (sessions, captures, logs, LLM calls)
├── prefs.json              # Sound/TTS preferences
├── spotify_token.json      # Cached Spotify OAuth token
├── fitbit_token.json       # Cached Fitbit OAuth token
└── piper-voices/           # Auto-downloaded Piper TTS voice model
```
