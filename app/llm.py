"""Optional Ollama assist for one-turn extraction and general replies."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Runtime status powers the small AI banner in the frontend.
_runtime_status: dict[str, Any] = {
    "enabled": False,
    "provider": None,
    "model": None,
    "last_ok": False,
    "last_error_code": None,
    "last_error_message": None,
    "quota_exceeded": False,
}
_ollama_probe_cache: dict[str, Any] = {"ts": 0.0, "ok": False}


class TurnAnalysis(BaseModel):
    """Model output for one caller utterance."""

    model_config = ConfigDict(extra="ignore")

    scheduling_intent: str | None = Field(
        default=None,
        description="book | reschedule when the user is clearly starting or continuing that flow",
    )
    patient_name: str | None = None
    appointment_date: str | None = Field(default=None, description="YYYY-MM-DD if known")
    appointment_time: str | None = Field(default=None, description="HH:MM 24h if known")
    old_appointment_date: str | None = Field(default=None, description="YYYY-MM-DD for reschedule")
    general_reply: str | None = Field(
        default=None,
        description="Brief helpful reply for informational questions only",
    )


def _ollama_base_url() -> str:
    # Returns base URL for local Ollama API.
    return (os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip().rstrip("/")


def _ollama_model() -> str:
    # Returns the configured Ollama model name.
    return (os.environ.get("OLLAMA_MODEL") or "llama3.2:3b").strip()


def _ollama_timeout_seconds() -> float:
    # Parses request timeout with a safe minimum.
    raw = (os.environ.get("OLLAMA_TIMEOUT_SECONDS") or "90").strip()
    try:
        value = float(raw)
    except ValueError:
        return 90.0
    return max(5.0, value)


def _ollama_keep_alive() -> str:
    # Returns model keep-alive duration for faster repeat turns.
    return (os.environ.get("OLLAMA_KEEP_ALIVE") or "30m").strip()


def llm_provider() -> str | None:
    # Resolves whether Ollama assist is on or intentionally disabled.
    """We keep provider logic simple: this project uses Ollama only."""
    override = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if override in ("none", "off", "disabled", "false", "0"):
        return None
    if override and override != "ollama":
        logger.warning("Unsupported LLM_PROVIDER=%s (expected 'ollama' or disabled); using ollama", override)
    return "ollama"


def _ollama_available() -> bool:
    # Quick health check so we can skip LLM calls when Ollama is down.
    """Fast health probe to avoid long request timeouts when Ollama is down."""
    now = time.time()
    # tiny cache so we do not ping /api/tags on every single chat turn
    if now - float(_ollama_probe_cache.get("ts", 0.0)) < 5.0:
        return bool(_ollama_probe_cache.get("ok", False))
    ok = False
    try:
        req = urllib.request.Request(f"{_ollama_base_url()}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            ok = 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        ok = False
    _ollama_probe_cache["ts"] = now
    _ollama_probe_cache["ok"] = ok
    return ok


def llm_enabled() -> bool:
    # Indicates whether LLM assist should be used for this run.
    p = llm_provider()
    if p == "ollama":
        return _ollama_available()
    return p is not None


def llm_model_label() -> str:
    # Returns model label used in status endpoints/UI.
    p = llm_provider()
    if p == "ollama":
        return _ollama_model()
    return ""


def get_llm_runtime_status() -> dict[str, Any]:
    # Exposes runtime status consumed by frontend AI banner.
    p = llm_provider()
    m = llm_model_label() or None
    return {
        "enabled": bool(p),
        "provider": p,
        "model": m,
        "last_ok": bool(_runtime_status.get("last_ok")),
        "last_error_code": _runtime_status.get("last_error_code"),
        "last_error_message": _runtime_status.get("last_error_message"),
        "quota_exceeded": bool(_runtime_status.get("quota_exceeded")),
    }


def _mark_llm_ok(provider: str, model: str) -> None:
    # Stores last successful LLM call info.
    _runtime_status.update(
        {
            "enabled": True,
            "provider": provider,
            "model": model,
            "last_ok": True,
            "last_error_code": None,
            "last_error_message": None,
            "quota_exceeded": False,
        }
    )


def _mark_llm_error(provider: str, model: str, err: Exception) -> None:
    # Stores last LLM error info for UI visibility and fallback behavior.
    msg = str(err)
    code = getattr(err, "status_code", None)
    if code is None:
        m = re.search(r"\b(4\d\d|5\d\d)\b", msg)
        code = int(m.group(1)) if m else None
    quota = "quota" in msg.lower() or "rate limit" in msg.lower() or code == 429
    _runtime_status.update(
        {
            "enabled": True,
            "provider": provider,
            "model": model,
            "last_ok": False,
            "last_error_code": code,
            "last_error_message": msg[:400],
            "quota_exceeded": quota,
        }
    )


def _strip_json_fence(raw: str) -> str:
    # Removes markdown code fences around model JSON when present.
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _system_prompt() -> str:
    # Defines structured extraction contract for the turn-level LLM call.
    return (
        "You help a medical practice phone reception assistant. "
        "You never diagnose, prescribe, or give clinical advice. "
        "For emergencies, tell them to hang up and call 911. "
        "Keep general_reply to at most three short sentences, plain text, no markdown.\n\n"
        "Return a single JSON object with these keys (use null when unknown):\n"
        '- scheduling_intent: null or "book" or "reschedule" — only when the caller clearly wants that.\n'
        "- patient_name: a real person's name only if clearly stated (e.g. 'for Jane Smith'). "
        "Never put a full booking sentence, question, or date phrase into patient_name.\n"
        "- appointment_date, old_appointment_date: strict YYYY-MM-DD if you can infer a calendar date.\n"
        "- appointment_time: HH:MM 24-hour if they gave a time.\n"
        "- general_reply: When dialog_state.phase is idle and the caller asks an informational question "
        "(office days or hours, parking, forms, insurance, what appointment types or services exist, "
        "policies, costs in general terms), set general_reply to a helpful reception-style answer. "
        "If they are clearly booking or rescheduling in that message, set general_reply to null.\n"
        "When phase is not idle, general_reply should almost always be null unless the message is only "
        "small talk or a pure FAQ with no scheduling content.\n\n"
        "Prefer extracting dates relative to today_utc. Do not invent patient names."
    )


def _user_payload(state_snapshot: dict[str, Any], user_text: str) -> str:
    # Serializes dialog state + caller message into one JSON payload.
    return json.dumps(
        {"dialog_state": state_snapshot, "caller_message": user_text},
        ensure_ascii=False,
    )


def _fetch_ollama(state_snapshot: dict[str, Any], user_text: str) -> TurnAnalysis | None:
    # Executes one non-streaming Ollama chat request and validates JSON response.
    base = _ollama_base_url()
    model = _ollama_model()
    system = _system_prompt()
    payload = _user_payload(state_snapshot, user_text)
    timeout_s = _ollama_timeout_seconds()

    body = {
        "model": model,
        "stream": False,
        "format": "json",
        # keep the model warm so responses stay fast in back-to-back turns
        "keep_alive": _ollama_keep_alive(),
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ],
    }
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            content = (obj.get("message") or {}).get("content", "").strip()
            if not content:
                raise RuntimeError("Ollama returned empty message content")
            data = json.loads(_strip_json_fence(content))
            _mark_llm_ok("ollama", model)
            return TurnAnalysis.model_validate(data)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="ignore")
        err = RuntimeError(f"{e.code} {msg[:300]}")
        _mark_llm_error("ollama", model, err)
        logger.warning("Ollama turn analysis failed: %s", err)
        return None
    except Exception as e:
        _mark_llm_error("ollama", model, e)
        logger.warning("Ollama turn analysis failed: %s", e)
        return None


def fetch_turn_analysis(
    state_snapshot: dict[str, Any],
    user_text: str,
) -> TurnAnalysis | None:
    # Public helper used by conversation engine to request structured turn hints.
    # If someone disables LLM in env, we cleanly skip assist.
    if llm_provider() is None:
        return None
    return _fetch_ollama(state_snapshot, user_text)
