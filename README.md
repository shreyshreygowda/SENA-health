# SENA Practice Voice Assistant

Prototype reception assistant for a medical practice: patients can **book** or **reschedule** appointments through **voice or text**, with multi-step clarification, light natural-language dates, and correction phrases such as “actually, next Monday.” Completed actions are **logged** to the console and to `logs/conversations.jsonl`.

## Quick start

Python **3.10+** recommended.

### 1) Install Ollama

#### macOS

Option A (official app):
1. Go to [ollama.com/download](https://ollama.com/download)
2. Download the macOS installer and install it

Option B (Homebrew):

```bash
brew install --cask ollama
```

#### Windows

1. Go to [ollama.com/download](https://ollama.com/download)
2. Download the Windows installer and run it
3. Open a new PowerShell after install completes

### 2) Pull the model once

```bash
ollama pull llama3.2:3b
```

### 3) Start Ollama service (Terminal 1)

```bash
ollama serve
```

Leave this terminal running.

### 4) Set up and run the app (Terminal 2)

#### macOS / Linux

```bash
cd SENA-health
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
lsof -ti :8765 | xargs kill -9 2>/dev/null
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2:3b
export OLLAMA_TIMEOUT_SECONDS=120
export OLLAMA_KEEP_ALIVE=30m
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

#### Windows (PowerShell)

```powershell
cd SENA-health
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.2:3b"
$env:OLLAMA_TIMEOUT_SECONDS="120"
$env:OLLAMA_KEEP_ALIVE="30m"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

### 5) Open the app

Open **http://127.0.0.1:8765** in Chrome or Edge.  
Allow microphone access for push-to-talk, or type in the text box.

### Text-only REPL

```bash
python -m app.cli
```

Use `/reset` to clear context and `/quit` to exit.

## Optional AI assist (Ollama local only)

Each turn can add a small **LLM pass** (strict JSON) to infer scheduling intent, fill messy **name / date / time**, and answer **general reception** questions briefly (no diagnosis; emergencies → 911). The **state machine** still owns confirmations and `FINAL_RESULT` logging.

### Ollama local (no API key)

This project uses local **Ollama** at `http://127.0.0.1:11434`.

Optional:
- `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
- `OLLAMA_TIMEOUT_SECONDS` (default `90`; increase on first model load)
- `OLLAMA_KEEP_ALIVE` (default `30m`; keeps model warm between turns)
- `LLM_PROVIDER=ollama` to force local even when cloud keys are set.

If Ollama is unavailable, the app falls back to rules. **`GET /api/features`** returns `llm_configured`, `llm_provider`, and `llm_model`.

## Technical approach

- **Conversation model**: A small deterministic **state machine** tracks intent (`book`, `reschedule`, `general`) and phase (collect name, dates, time). This keeps behavior predictable for a front-desk workflow while still feeling conversational.
- **Understanding**: **Keyword routing** selects the intent; **dateparser** resolves relative dates (“next Friday”); lightweight **regex** extracts times and common name patterns (“my name is …”). **Correction detection** watches for cues like “actually” or “I meant” during slot filling so users can revise a date or time without restarting. With **Ollama** running, a small **LLM pass** augments extraction and general answers; validated fields are merged so the state machine still owns booking and final confirmation.
- **Voice**: The browser **Web Speech API** handles speech-to-text and speech synthesis, so the demo runs without native audio drivers or cloud keys. The backend stays simple JSON over HTTP.
- **Actions**: Booking and rescheduling **simulate** a practice system: the engine prints a structured `FINAL_RESULT` line and appends JSON lines to `logs/conversations.jsonl` for traceability.

## API (for integrations)

| Method | Path           | Body                                      | Notes                |
| ------ | -------------- | ----------------------------------------- | -------------------- |
| POST   | `/api/session` | —                                         | Returns `session_id` |
| POST   | `/api/chat`    | `{"session_id":"…","message":"…"}`       | Assistant reply      |
| POST   | `/api/reset`   | `{"session_id":"…","message":""}`         | Clears dialog state  |
| GET    | `/api/features` | —                                        | `llm_configured`, `llm_provider`, `llm_model` |

## Tests

```bash
pip install pytest
pytest -q
```

## Submission

Send a repository link or archive to **poosthuizen@senahealth.com** as described in the challenge brief.
