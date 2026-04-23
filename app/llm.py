"""Optional LLM assist: Google Gemini or OpenAI — structured extraction + general reception answers."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_runtime_status: dict[str, Any] = {
    "enabled": False,
    "provider": None,
    "model": None,
    "last_ok": False,
    "last_error_code": None,
    "last_error_message": None,
    "quota_exceeded": False,
}


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


def _openai_key() -> str | None:
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    return k or None


def _gemini_key() -> str | None:
    k = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    return k or None


def llm_provider() -> Literal["openai", "gemini"] | None:
    """Default: Gemini if `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set, else OpenAI. Override with `LLM_PROVIDER=openai` or `gemini`."""
    override = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if override == "openai" and _openai_key():
        return "openai"
    if override == "gemini" and _gemini_key():
        return "gemini"
    if _gemini_key():
        return "gemini"
    if _openai_key():
        return "openai"
    return None


def llm_enabled() -> bool:
    return llm_provider() is not None


def llm_model_label() -> str:
    p = llm_provider()
    if p == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    if p == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    return ""


def get_llm_runtime_status() -> dict[str, Any]:
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
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _system_prompt() -> str:
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
    return json.dumps(
        {"dialog_state": state_snapshot, "caller_message": user_text},
        ensure_ascii=False,
    )


def _fetch_openai(state_snapshot: dict[str, Any], user_text: str) -> TurnAnalysis | None:
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; cannot use OpenAI assist")
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    system = _system_prompt()
    payload = _user_payload(state_snapshot, user_text)

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(_strip_json_fence(raw))
        _mark_llm_ok("openai", model)
        return TurnAnalysis.model_validate(data)
    except Exception as e:
        _mark_llm_error("openai", model, e)
        logger.warning("OpenAI turn analysis failed: %s", e)
        return None


def _fetch_gemini(state_snapshot: dict[str, Any], user_text: str) -> TurnAnalysis | None:
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google-generativeai not installed; cannot use Gemini assist")
        return None

    api_key = _gemini_key()
    if not api_key:
        return None

    model_id = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    system = _system_prompt()
    payload = _user_payload(state_snapshot, user_text)
    combined = f"{system}\n\nRespond with JSON only, no other text.\n\n{payload}"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_id,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
        response = model.generate_content(combined)
        raw = (getattr(response, "text", None) or "").strip()
        if not raw and response.candidates:
            parts = response.candidates[0].content.parts
            raw = "".join(getattr(p, "text", "") or "" for p in parts).strip()
        if not raw:
            logger.warning("Gemini returned empty response")
            return None
        data = json.loads(_strip_json_fence(raw))
        _mark_llm_ok("gemini", model_id)
        return TurnAnalysis.model_validate(data)
    except Exception as e:
        _mark_llm_error("gemini", model_id, e)
        logger.warning("Gemini turn analysis failed: %s", e)
        return None


def fetch_turn_analysis(
    state_snapshot: dict[str, Any],
    user_text: str,
) -> TurnAnalysis | None:
    provider = llm_provider()
    if provider == "gemini":
        return _fetch_gemini(state_snapshot, user_text)
    if provider == "openai":
        return _fetch_openai(state_snapshot, user_text)
    return None
