# SENA Practice Voice Assistant

Prototype reception assistant for a medical practice: patients can **book** or **reschedule** appointments through **voice or text**, with multi-step clarification, light natural-language dates, and correction phrases such as “actually, next Monday.” Completed actions are **logged** to the console and to `logs/conversations.jsonl`.

## Quick start

Python **3.10+** recommended.

```bash
cd SENA-health
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: export GEMINI_API_KEY=...   # Google AI Studio — enables model assist (see below)
# optional: export OPENAI_API_KEY=...   # alternative LLM provider
uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

Open **http://127.0.0.1:8765** in Chrome or Edge. Allow the microphone for push-to-talk; you can always type instead.

### Text-only REPL

```bash
python -m app.cli
```

Use `/reset` to clear context and `/quit` to exit.

## Optional AI assist (Gemini or OpenAI)

With an API key, each turn adds a small **LLM pass** (strict JSON) to infer scheduling intent, fill messy **name / date / time**, and answer **general reception** questions briefly (no diagnosis; emergencies → 911). The **state machine** still owns confirmations and `FINAL_RESULT` logging.

### Google Gemini (recommended if you do not use OpenAI)

1. Create a key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Set **`GEMINI_API_KEY`** (or **`GOOGLE_API_KEY`**) in your environment.
3. Optional: **`GEMINI_MODEL`** (default `gemini-2.0-flash`; try `gemini-1.5-flash` if your project does not have 2.0 yet).

If both Gemini and OpenAI keys are set, **Gemini is used by default**. Set **`LLM_PROVIDER=openai`** to force OpenAI instead.

### OpenAI

Set **`OPENAI_API_KEY`**. Optional: **`OPENAI_MODEL`** (default `gpt-4o-mini`).

Without any key, behavior is unchanged: rules, **dateparser**, and local date patterns only. **`GET /api/features`** returns `llm_configured`, `llm_provider`, and `llm_model`.

## Technical approach

- **Conversation model**: A small deterministic **state machine** tracks intent (`book`, `reschedule`, `general`) and phase (collect name, dates, time). This keeps behavior predictable for a front-desk workflow while still feeling conversational.
- **Understanding**: **Keyword routing** selects the intent; **dateparser** resolves relative dates (“next Friday”); lightweight **regex** extracts times and common name patterns (“my name is …”). **Correction detection** watches for cues like “actually” or “I meant” during slot filling so users can revise a date or time without restarting. With **Gemini** or **OpenAI** configured, a small **LLM pass** augments extraction and general answers; validated fields are merged so the state machine still owns booking and final confirmation.
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
