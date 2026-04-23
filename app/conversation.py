from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import dateparser

from app.llm import TurnAnalysis, fetch_turn_analysis, get_llm_runtime_status

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    BOOK = "book"
    RESCHEDULE = "reschedule"
    GENERAL = "general"


class Phase(str, Enum):
    IDLE = "idle"
    COLLECT_NAME = "collect_name"
    COLLECT_DATE = "collect_date"
    COLLECT_TIME = "collect_time"
    COLLECT_OLD_DATE = "collect_old_date"
    COLLECT_NEW_DATE = "collect_new_date"
    CONFIRM = "confirm"
    DONE = "done"


CORRECTION_MARKERS = (
    "actually",
    "i meant",
    "sorry",
    "change",
    "instead",
    "make it",
    "rather",
    "correction",
    "wait",
    "no ",
    "not ",
)

_MONTH_NUM: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass
class ConversationState:
    intent: Intent | None = None
    phase: Phase = Phase.IDLE
    patient_name: str | None = None
    appointment_date: str | None = None  # ISO date YYYY-MM-DD
    appointment_time: str | None = None  # HH:MM
    old_appointment_date: str | None = None
    transcript: list[dict[str, str]] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value if self.intent else None,
            "phase": self.phase.value,
            "patient_name": self.patient_name,
            "appointment_date": self.appointment_date,
            "appointment_time": self.appointment_time,
            "old_appointment_date": self.old_appointment_date,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _looks_like_correction(text: str) -> bool:
    t = _normalize(text)
    return any(m in t for m in CORRECTION_MARKERS)


_NAME_TOKEN_REJECT = frozenset(
    {
        "next",
        "this",
        "last",
        "appointment",
        "book",
        "schedule",
        "wait",
        "actually",
        "can",
        "could",
        "would",
        "for",
        "the",
        "my",
        "it",
        "do",
        "on",
        "at",
        "may",
        "june",
        "march",
        "april",
        "july",
        "january",
        "february",
        "august",
        "september",
        "october",
        "november",
        "december",
        "tomorrow",
        "today",
        "tonight",
        "soon",
        "later",
    }
)


def _name_tokens_plausible(name: str) -> bool:
    for w in name.split():
        wl = re.sub(r"[^a-z]", "", w.lower())
        if not wl:
            continue
        if wl in _NAME_TOKEN_REJECT or w.lower().endswith("day"):
            return False
    return True


def _parse_relative_weekday_phrase(text: str, ref: datetime) -> date | None:
    """dateparser often misses 'next Thursday'; resolve relative weekdays explicitly."""
    t = _normalize(text)
    wd_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    hits = list(
        re.finditer(
            r"(?i)\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            t,
        )
    )
    if hits:
        m = hits[-1]
        qual, day = m.group(1).lower(), m.group(2).lower()
    else:
        bare = list(re.finditer(r"(?i)\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t))
        if not bare:
            return None
        m = bare[-1]
        qual, day = "this", m.group(1).lower()
    tgt = wd_map[day]
    cur_d = ref.date() if isinstance(ref, datetime) else ref
    cur_wd = cur_d.weekday()
    delta = (tgt - cur_wd) % 7
    if delta == 0 and qual == "next":
        delta = 7
    return cur_d + timedelta(days=delta)


def _extract_name(text: str) -> str | None:
    """Lightweight name capture from common phrasings."""
    raw = text.strip()
    patterns = (
        r"(?:my name is|i'?m|i am|this is|it'?s|call me|name is)\s+([A-Za-z][A-Za-z\s'-]{1,40})",
        r"(?:for|under the name)\s+([A-Za-z][A-Za-z\s'-]{1,40})",
        r"(?:book|schedule)\s+(?:an\s+)?(?:appointment|visit)\s+for\s+([A-Za-z][A-Za-z'-]{1,32})\b",
    )
    for pat in patterns:
        m = re.search(pat, raw, re.I)
        if m:
            name = m.group(1).strip()
            # Trim trailing filler words
            name = re.split(r"\b(and|for|on|at|the|a)\b", name, maxsplit=1, flags=re.I)[0].strip()
            if 1 < len(name) < 60 and _name_tokens_plausible(name):
                return name.title()
    return _fallback_name_from_line(raw)


_NAME_BLOCKLIST = frozenset(
    {
        "appointment",
        "book",
        "schedule",
        "reschedule",
        "wednesday",
        "thursday",
        "friday",
        "monday",
        "tuesday",
        "saturday",
        "sunday",
        "next",
        "last",
        "this",
        "can",
        "could",
        "would",
        "wait",
        "actually",
        "please",
        "the",
        "for",
        "june",
        "may",
        "march",
        "april",
        "july",
        "august",
        "january",
        "february",
        "september",
        "october",
        "november",
        "december",
        "want",
        "need",
        "help",
        "my",
        "an",
        "pm",
        "am",
        "yes",
        "no",
        "ok",
        "hi",
        "hey",
        "hello",
        "thanks",
        "thank",
    }
)


def _fallback_name_from_line(text: str) -> str | None:
    """Single-line first/last name only — never the whole booking sentence."""
    raw = text.strip()
    if not raw or len(raw) > 40:
        return None
    words = raw.split()
    if not words or len(words) > 3:
        return None
    for w in words:
        wl = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", w).lower()
        if not wl:
            continue
        if wl in _NAME_BLOCKLIST or wl.endswith("day"):
            return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*", raw):
        return None
    return raw.title()


def _strip_leading_date_filler(text: str) -> str:
    """Remove polite / hedging words so 'can I please do May 6th' yields a parseable tail."""
    s = text.strip()
    # Greedy strip of common prefixes (repeat to peel "can i please can i...")
    for _ in range(4):
        nxt = re.sub(
            r"(?i)^(can|could|would|may)\s+i\s+(?:please\s+)?(?:like\s+to\s+)?"
            r"(?:do|get|have|book|schedule|use|pick|take|make\s+it|go\s+with)\s+",
            "",
            s,
        )
        nxt = re.sub(r"(?i)^(please|thanks|ok|okay|yes|well)\s*[,.]?\s*", "", nxt)
        nxt = re.sub(r"(?i)^(i\s+(?:want|need|was\s+hoping))\s+(?:to\s+)?(?:do|get|have|book)?\s*", "", nxt)
        if nxt == s:
            break
        s = nxt.strip()
    return s


def _ordinals_to_cardinals(s: str) -> str:
    return re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)


def _date_substrings_to_try(text: str) -> list[str]:
    """Prefer the clause after a self-correction ('… actually next Thursday')."""
    raw = text.strip()
    parts = re.split(r"(?i)\b(?:actually|wait|i meant|instead)\b\s*,?\s*", raw)
    parts = [p.strip() for p in parts if p.strip()]
    ordered: list[str] = []
    if len(parts) >= 2:
        ordered.append(parts[-1])
    if raw not in ordered:
        ordered.append(raw)
    return ordered


def _parse_ordinal_day_of_month(text: str, base: datetime) -> date | None:
    """Phrases like 'the 10th of may' or '5th of june 2027'."""
    s = _ordinals_to_cardinals(_normalize(text))
    month_alt = "|".join(sorted(_MONTH_NUM.keys(), key=len, reverse=True))
    m = re.search(
        rf"(?i)\b(?:the\s+)?(\d{{1,2}})\s+of\s+({month_alt})\b(?:\s*,?\s*(\d{{4}}))?",
        s,
    )
    if not m:
        return None
    day = int(m.group(1))
    mon_word = m.group(2).lower()
    month = _MONTH_NUM.get(mon_word)
    if not month:
        return None
    year_s = m.group(3)
    base_d = base.date() if isinstance(base, datetime) else base
    year = int(year_s) if year_s else base_d.year
    try:
        candidate = date(year, month, day)
    except ValueError:
        return None
    if not year_s and candidate < base_d:
        try:
            candidate = date(year + 1, month, day)
        except ValueError:
            return None
    return candidate


def _parse_month_name_date(text: str, base: datetime) -> date | None:
    """
    Explicit US-style 'May 6 2027', 'Thursday May 17', 'may 6th 2027' (dateparser often misses ordinals).
    Longer month names first so 'september' beats 'sep'.
    """
    s = _ordinals_to_cardinals(_normalize(text))
    s = re.sub(
        r"(?i)^(monday|tuesday|wednesday|thursday|friday|saturday|sun|mon|tues|tue|wed|thur|thu|fri|sat)\b\s*,?\s*",
        "",
        s,
    )
    months: tuple[tuple[str, int], ...] = (
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("sept", 9),
        ("sep", 9),
        ("oct", 10),
        ("nov", 11),
        ("dec", 12),
        ("jan", 1),
        ("feb", 2),
        ("mar", 3),
        ("apr", 4),
        ("jun", 6),
        ("jul", 7),
        ("aug", 8),
        ("may", 5),
    )
    base_d = base.date() if isinstance(base, datetime) else base
    for name, month in months:
        m = re.search(
            rf"(?i)\b{re.escape(name)}\s+(\d{{1,2}})\b(?:\s*,?\s*(\d{{4}}))?(?!\s*[ap]m)\b",
            s,
        )
        if not m:
            continue
        day = int(m.group(1))
        year_s = m.group(2)
        year = int(year_s) if year_s else base_d.year
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if not year_s and candidate < base_d:
            try:
                candidate = date(year + 1, month, day)
            except ValueError:
                continue
        return candidate
    return None


def _parse_one_date_chunk(chunk: str, ref: datetime) -> str | None:
    raw = chunk.strip()
    lower = raw.lower()
    for prefix in ("actually,", "actually", "sorry,", "sorry", "i meant", "make it", "instead,"):
        if lower.startswith(prefix):
            raw = raw[len(prefix) :].lstrip(" ,.-")
            lower = raw.lower()
            break

    cleaned = _strip_leading_date_filler(raw)
    od = _parse_ordinal_day_of_month(cleaned, ref)
    if od:
        return od.isoformat()
    explicit = _parse_month_name_date(cleaned, ref)
    if explicit:
        return explicit.isoformat()
    rw = _parse_relative_weekday_phrase(cleaned, ref)
    if rw:
        return rw.isoformat()

    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": ref,
        "DATE_ORDER": "MDY",
    }
    for candidate in (_ordinals_to_cardinals(cleaned), cleaned):
        dt = dateparser.parse(candidate, settings=settings)
        if dt:
            return dt.date().isoformat()
    return None


def _parse_date_from_text(text: str, base: datetime | None = None) -> str | None:
    ref = base or datetime.now(UTC)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=UTC)
    for chunk in _date_substrings_to_try(text):
        got = _parse_one_date_chunk(chunk, ref)
        if got:
            return got
    return None


def _parse_time_from_text(text: str) -> str | None:
    t = _normalize(text)
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        h, minute, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if ap == "pm" and h != 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        return f"{h:02d}:{minute:02d}"
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    if "noon" in t or "midday" in t:
        return "12:00"
    if "morning" in t and re.search(r"\b\d", t) is None:
        return "09:00"
    if "afternoon" in t and re.search(r"\b\d", t) is None:
        return "14:00"
    return None


def _time_correction_segments(text: str) -> list[str]:
    raw = text.strip()
    parts = re.split(r"(?i)\b(?:actually|wait|i meant|instead|no)\b\s*,?\s*", raw)
    return [p.strip() for p in parts if p.strip()]


def _bare_hour_default_pm(hour: int) -> str | None:
    """Spoken 'seven' / 'ten' without a.m.–p.m. during booking → assume evening slots."""
    if not (1 <= hour <= 12):
        return None
    if hour == 12:
        return "12:00"
    return f"{hour + 12:02d}:00"


def _parse_time_from_utterance(text: str) -> str | None:
    """Try full line, then the last self-correction clause, then a trailing bare hour."""
    segs = _time_correction_segments(text)
    ordered: list[str] = []
    if len(segs) >= 2:
        ordered.append(segs[-1])
    ordered.append(text.strip())
    for seg in ordered:
        tm = _parse_time_from_text(seg)
        if tm:
            return tm
    for seg in ordered:
        t = _normalize(seg)
        if "morning" in t and "afternoon" not in t:
            m = re.search(r"\b(\d{1,2})\s*(am|pm)?\s*$", t)
            if m and not m.group(2):
                h = int(m.group(1))
                if 1 <= h <= 11:
                    return f"{h:02d}:00"
        m = re.search(r"(?:^|\s)(?:do\s+)?(\d{1,2})\s*(am|pm)\s*$", t)
        if m:
            h, ap = int(m.group(1)), m.group(2).lower()
            if ap == "pm" and h != 12:
                h += 12
            if ap == "am" and h == 12:
                h = 0
            return f"{h:02d}:00"
        m2 = re.search(r"(?:^|\s)(?:do\s+)?(\d{1,2})\s*$", t)
        if m2 and not re.search(r"\b(am|pm)\b", t):
            return _bare_hour_default_pm(int(m2.group(1)))
    return None


def _utterance_has_time_signal(text: str) -> bool:
    t = _normalize(text)
    if re.search(r"\b\d{1,2}\s*(am|pm)\b", t):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\b", t):
        return True
    if re.search(r"(?:^|\s)do\s+\d{1,2}\b", t):
        return True
    if _looks_like_correction(text) and re.search(r"(?:^|\s)(?:do\s+)?(\d{1,2})\s*$", t):
        return True
    return False


def _utterance_has_date_signal(text: str) -> bool:
    """Avoid treating '7 … 10' as a calendar change while collecting time."""
    t = _normalize(text)
    if re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        t,
    ):
        return True
    if re.search(r"\b(tomorrow|today|tonight)\b", t):
        return True
    if re.search(r"\b(next|this)\s+(mon|tues|tue|wed|thu|thur|fri|sat|sun)", t):
        return True
    if re.search(r"\d{1,2}(?:st|nd|rd|th)\s+of\s+", t):
        return True
    if re.search(r"\b20\d{2}\b", t):
        return True
    if re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        t,
    ):
        return True
    return False


def _coerce_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    s = value.strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return None
    y, mo, d = map(int, s.split("-"))
    try:
        date(y, mo, d)
    except ValueError:
        return None
    return s


def _coerce_time(value: str | None) -> str | None:
    if not value:
        return None
    s = value.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return f"{h:02d}:{mi:02d}"


def _llm_patient_name(hint: TurnAnalysis | None) -> str | None:
    if not hint or not hint.patient_name:
        return None
    n = hint.patient_name.strip()
    if len(n) < 2 or len(n) > 60 or re.search(r"\d{3,}", n):
        return None
    if not _name_tokens_plausible(n):
        return None
    return n.title()


def _embedded_relative_day_word(text: str) -> str | None:
    t = _normalize(text)
    for w in ("tomorrow", "today", "tonight"):
        if re.search(rf"(?i)\b{re.escape(w)}\b", t):
            return w
    return None


def classify_intent(text: str) -> Intent:
    t = _normalize(text)
    reschedule_signals = (
        "reschedule",
        "move my appointment",
        "change my appointment",
        "different day",
        "another day",
        "can't make",
        "cant make",
        "push my appointment",
    )
    book_signals = (
        "book",
        "schedule",
        "make an appointment",
        "new appointment",
        "see dr",
        "see doctor",
        "visit",
        "come in",
        "availability",
    )
    if any(s in t for s in reschedule_signals):
        return Intent.RESCHEDULE
    if any(s in t for s in book_signals):
        return Intent.BOOK
    return Intent.GENERAL


_DOCTOR_FAQ_REPLY = (
    "I can't see your chart from this line. Your appointment confirmation or patient portal lists the "
    "provider, or the front desk can look it up when you arrive."
)


def _is_doctor_faq(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(what|which)\s+doctor\b|\bwho\s+(am i|will i)\s+see\b|\bwho\s+is\s+my\s+doctor\b|"
            r"what\s+doctor.*\b(see|seeing)\b|which\s+doctor.*\b(see|seeing)\b",
            _normalize(text),
        )
    )


GENERAL_RESPONSES: list[tuple[str, str]] = [
    (
        r"what doctor|which doctor|who (am i|will i) see|who is my doctor|what physician|"
        r"provider (for|am i)|doctor am i|what doctor.*(see|seeing)|which doctor.*(see|seeing)",
        _DOCTOR_FAQ_REPLY,
    ),
    (
        r"what days|which days|days (is|are)|office (hours|days)|when.*open|open.*week|"
        r"open (on|for) weekdays|weekday hours",
        "We're open Monday through Friday, 8 a.m. to 5 p.m.",
    ),
    (r"hour|open|close|when are you", "We're open Monday through Friday, 8 a.m. to 5 p.m."),
    (
        r"type of appointment|kinds of appointment|what appointment|appointments do i|"
        r"what can i (book|schedule)|services do you offer|what visits",
        "We offer routine visits, follow-up visits, and preventive care. Your provider decides the visit type you need.",
    ),
    (r"insurance|accept|coverage", "We accept most major insurance plans. Bring your card to check benefits."),
    (r"address|location|where|park", "We're at 1200 Wellness Way, Suite 300. Visitor parking is on level B."),
    (r"emergency|urgent|severe pain|chest pain", "If this is an emergency, hang up and dial nine one one."),
    (r"cost|price|fee|pay", "Costs depend on your visit type and insurance. Billing can give a ballpark after scheduling."),
    (r"thank|thanks", "You're welcome. Is there anything else I can help with?"),
    (
        r"(?i)^(hello|hey)\b|^(hi)(\s*[,.!])?\s*$",
        "Hello. I can help you book or reschedule an appointment, or answer general questions.",
    ),
]


_GENERAL_FALLBACK = (
    "I can help you book a new appointment, reschedule an existing one, or answer questions about "
    "hours and insurance. What would you like to do?"
)


def match_canned_general(text: str) -> str | None:
    t = _normalize(text)
    for pattern, reply in GENERAL_RESPONSES:
        if re.search(pattern, t):
            return reply
    return None


def general_reply(text: str) -> str:
    return match_canned_general(text) or _GENERAL_FALLBACK


class ConversationEngine:
    """Multi-turn slot filling with correction handling."""

    def __init__(self, log_path: Path | None = None, enable_llm: bool | None = None) -> None:
        self.state = ConversationState()
        self.log_path = log_path
        if enable_llm is None:
            from app.llm import llm_enabled

            enable_llm = llm_enabled()
        self.enable_llm = enable_llm

    def _llm_context(self) -> dict[str, Any]:
        return {
            "phase": self.state.phase.value,
            "intent": self.state.intent.value if self.state.intent else None,
            "patient_name": self.state.patient_name,
            "appointment_date": self.state.appointment_date,
            "appointment_time": self.state.appointment_time,
            "old_appointment_date": self.state.old_appointment_date,
            "recent_transcript": self.state.transcript[-10:],
            "today_utc": datetime.now(UTC).date().isoformat(),
        }

    def _date_from_text_or_llm(self, text: str, hint: TurnAnalysis | None) -> str | None:
        d = _parse_date_from_text(text)
        if d:
            return d
        if hint and hint.appointment_date:
            return _coerce_iso_date(hint.appointment_date)
        return None

    def _date_from_chatty_booking_line(self, text: str, hint: TurnAnalysis | None) -> str | None:
        """Pick up 'tomorrow' etc. after filler like 'book an appointment for …'."""
        d = self._date_from_text_or_llm(text, hint)
        if d:
            return d
        rel = _embedded_relative_day_word(text)
        if rel:
            d = self._date_from_text_or_llm(rel, hint)
            if d:
                return d
        m = re.search(r"(?i)\bfor\s+(.+)$", text.strip())
        if m:
            return self._date_from_text_or_llm(m.group(1).strip(), hint)
        return None

    def _old_date_from_text_or_llm(self, text: str, hint: TurnAnalysis | None) -> str | None:
        d = _parse_date_from_text(text)
        if d:
            return d
        if hint and hint.old_appointment_date:
            return _coerce_iso_date(hint.old_appointment_date)
        return None

    def _time_from_text_or_llm(self, text: str, hint: TurnAnalysis | None) -> str | None:
        tm = _parse_time_from_utterance(text)
        if tm:
            return tm
        if hint and hint.appointment_time:
            return _coerce_time(hint.appointment_time)
        return None

    def reset(self) -> None:
        self.state = ConversationState()

    def _log_turn(self, role: str, content: str) -> None:
        self.state.transcript.append(
            {"role": role, "content": content, "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z")}
        )
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n"
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def _finalize_booking(self) -> str:
        record = {
            "action": "BOOKED",
            "patient_name": self.state.patient_name,
            "date": self.state.appointment_date,
            "time": self.state.appointment_time or "09:00",
        }
        msg = (
            f"Booking confirmed for {self.state.patient_name} on {self.state.appointment_date} "
            f"at {record['time']}. You'll receive a reminder the day before."
        )
        logger.info("FINAL_RESULT %s", json.dumps(record))
        self._log_turn("system", f"FINAL_RESULT {json.dumps(record)}")
        self.state.phase = Phase.DONE
        return msg

    def _finalize_reschedule(self) -> str:
        record = {
            "action": "RESCHEDULED",
            "patient_name": self.state.patient_name,
            "old_date": self.state.old_appointment_date,
            "new_date": self.state.appointment_date,
            "time": self.state.appointment_time or "09:00",
        }
        old = self.state.old_appointment_date or "your prior visit"
        msg = (
            f"I've moved {self.state.patient_name}'s appointment from {old} to "
            f"{self.state.appointment_date} at {record['time']}."
        )
        logger.info("FINAL_RESULT %s", json.dumps(record))
        self._log_turn("system", f"FINAL_RESULT {json.dumps(record)}")
        self.state.phase = Phase.DONE
        return msg

    def process_message(self, user_text: str) -> tuple[str, dict[str, Any]]:
        text = user_text.strip()
        if not text:
            out = self.state.to_public_dict()
            out["done"] = False
            out["ai_hint"] = False
            out["ai_status"] = get_llm_runtime_status()
            return "I didn't catch that. Could you repeat that?", out

        hint: TurnAnalysis | None = None
        if self.enable_llm:
            try:
                hint = fetch_turn_analysis(self._llm_context(), text)
            except Exception:
                logger.exception("LLM assist failed; continuing with rules only")

        self._log_turn("user", text)
        reply, done = self._step(text, hint)
        self._log_turn("assistant", reply)
        payload = self.state.to_public_dict()
        payload["done"] = done
        payload["ai_hint"] = bool(hint) if self.enable_llm else False
        payload["ai_status"] = get_llm_runtime_status()
        return reply, payload

    def _step(self, text: str, hint: TurnAnalysis | None = None) -> tuple[str, bool]:
        s = self.state

        # Silence / unclear very short
        if len(text) < 2:
            return "I didn't quite understand. Could you say that again?", False

        # Fresh start after done
        if s.phase == Phase.DONE:
            self.reset()
            s = self.state

        if s.phase == Phase.IDLE:
            rule_intent = classify_intent(text)
            if rule_intent != Intent.GENERAL:
                s.intent = rule_intent
            elif hint:
                si = (hint.scheduling_intent or "").lower()
                if si in ("book", "reschedule"):
                    s.intent = Intent.BOOK if si == "book" else Intent.RESCHEDULE
                else:
                    s.intent = Intent.GENERAL
            else:
                s.intent = Intent.GENERAL

            if s.intent == Intent.GENERAL:
                canned = match_canned_general(text)
                if canned:
                    return canned, True
                if hint and hint.general_reply and hint.general_reply.strip():
                    return hint.general_reply.strip(), True
                return general_reply(text), True
            if s.intent == Intent.BOOK:
                s.phase = Phase.COLLECT_NAME
                name = _extract_name(text) or _llm_patient_name(hint)
                d0 = self._date_from_chatty_booking_line(text, hint)
                if name:
                    s.patient_name = name
                    s.phase = Phase.COLLECT_DATE
                    if d0:
                        s.appointment_date = d0
                        s.phase = Phase.COLLECT_TIME
                        return (
                            f"Thanks, {s.patient_name}. I have {d0}. Do you prefer morning or afternoon, "
                            "or a specific time?",
                            False,
                        )
                    return f"Thanks, {s.patient_name}. What day works best for your visit?", False
                if d0:
                    s.appointment_date = d0
                    s.phase = Phase.COLLECT_NAME
                    return (
                        f"I have {d0} on the calendar. Who should I book this visit for?",
                        False,
                    )
                return "I can help you book an appointment. What's the patient's full name?", False

            # RESCHEDULE
            s.phase = Phase.COLLECT_NAME
            name = _extract_name(text) or _llm_patient_name(hint)
            if name:
                s.patient_name = name
                s.phase = Phase.COLLECT_OLD_DATE
                return "What's the date of the appointment you'd like to move?", False
            return "I can reschedule that. Who is the appointment for?", False

        if s.phase not in (Phase.IDLE, Phase.DONE) and s.intent in (Intent.BOOK, Intent.RESCHEDULE):
            if _is_doctor_faq(text):
                return _DOCTOR_FAQ_REPLY, False

        # --- Active flows ---
        if s.intent == Intent.BOOK:
            return self._flow_book(text, hint)
        if s.intent == Intent.RESCHEDULE:
            return self._flow_reschedule(text, hint)
        return general_reply(text), True

    def _flow_book(self, text: str, hint: TurnAnalysis | None = None) -> tuple[str, bool]:
        s = self.state
        if s.phase == Phase.COLLECT_NAME:
            if _looks_like_correction(text) and s.patient_name:
                dfix = self._date_from_text_or_llm(text, hint)
                if dfix:
                    s.appointment_date = dfix
                    s.phase = Phase.COLLECT_TIME
                    return f"Updated to {s.appointment_date}. What time should I hold?", False
            name = _extract_name(text) or _llm_patient_name(hint)
            if not name or len(name) < 2:
                return "Could you tell me the name again?", False
            s.patient_name = name
            s.phase = Phase.COLLECT_DATE
            d = self._date_from_text_or_llm(text, hint)
            if d:
                s.appointment_date = d
                s.phase = Phase.COLLECT_TIME
                return (
                    f"Got it, {s.patient_name}. I have {d}. Morning, afternoon, or a specific time?",
                    False,
                )
            return f"Thanks, {s.patient_name}. Which date would you like?", False

        if s.phase == Phase.COLLECT_DATE:
            if _looks_like_correction(text):
                tfix = self._time_from_text_or_llm(text, hint)
                if tfix and not _utterance_has_date_signal(text) and s.appointment_date:
                    s.appointment_time = tfix
                    s.phase = Phase.COLLECT_TIME
                    return self._finalize_booking(), True
                if tfix and not _utterance_has_date_signal(text) and not s.appointment_date:
                    return (
                        "I heard a time, but I still need the visit day first — try tomorrow, next Friday, "
                        "or a date like May sixth.",
                        False,
                    )
                dfix = self._date_from_text_or_llm(text, hint) if _utterance_has_date_signal(text) else None
                if dfix:
                    s.appointment_date = dfix
                    s.phase = Phase.COLLECT_TIME
                    return f"Updated to {s.appointment_date}. What time works?", False
                return (
                    "I couldn't update that date. Try tomorrow, a weekday, or a month and day with the year.",
                    False,
                )
            d = self._date_from_text_or_llm(text, hint)
            if not d:
                return (
                    "I couldn't parse that date. Try a month and day with the year, for example May 6th 2027, "
                    "Thursday May 17th, or say next Tuesday."
                ), False
            s.appointment_date = d
            s.phase = Phase.COLLECT_TIME
            return "Great. Preferred time?", False

        if s.phase == Phase.COLLECT_TIME:
            if _looks_like_correction(text):
                tfix = self._time_from_text_or_llm(text, hint)
                if tfix:
                    s.appointment_time = tfix
                    return self._finalize_booking(), True
                if _utterance_has_date_signal(text):
                    dfix = self._date_from_text_or_llm(text, hint)
                    if dfix:
                        s.appointment_date = dfix
                        return f"Date updated to {s.appointment_date}. What time should I set?", False
                return (
                    "Say a time with a.m. or p.m., like seven p.m. or ten p.m., or tell me the calendar date "
                    "you want to switch to."
                ), False
            tm = self._time_from_text_or_llm(text, hint)
            tn = _normalize(text)
            if not tm and "morning" in tn:
                tm = "09:00"
            elif not tm and "afternoon" in tn:
                tm = "14:00"
            elif not tm and ("either" in tn or "any time" in tn or tn == "any"):
                tm = "10:00"
            if not tm:
                # allow skipping with "any time"
                if "any" in _normalize(text) or "doesn't matter" in _normalize(text):
                    tm = "10:00"
            if not tm:
                return "Say a time like ten thirty a.m., or say morning or afternoon.", False
            s.appointment_time = tm
            return self._finalize_booking(), True

        return "Let's start over. How can I help?", True

    def _flow_reschedule(self, text: str, hint: TurnAnalysis | None = None) -> tuple[str, bool]:
        s = self.state
        if s.phase == Phase.COLLECT_NAME:
            name = _extract_name(text) or _llm_patient_name(hint)
            if not name or len(name) < 2:
                return "Who should I look up?", False
            s.patient_name = name
            s.phase = Phase.COLLECT_OLD_DATE
            return "What's the current appointment date you want to move?", False

        if s.phase == Phase.COLLECT_OLD_DATE:
            d = self._old_date_from_text_or_llm(text, hint)
            if not d:
                return "Which date is the existing appointment on?", False
            s.old_appointment_date = d
            s.appointment_date = None
            s.phase = Phase.COLLECT_NEW_DATE
            return "What new date works for you?", False

        if s.phase == Phase.COLLECT_NEW_DATE:
            if _looks_like_correction(text):
                dfix = self._date_from_text_or_llm(text, hint)
                if dfix:
                    s.appointment_date = dfix
                    s.phase = Phase.COLLECT_TIME
                    return f"Updated to {s.appointment_date}. What time on that day?", False
            d = self._date_from_text_or_llm(text, hint)
            if not d:
                return "I didn't get the new date. For example, say June third or next Monday.", False
            s.appointment_date = d
            s.phase = Phase.COLLECT_TIME
            return "Preferred time for the new slot?", False

        if s.phase == Phase.COLLECT_TIME:
            if _looks_like_correction(text):
                tfix = self._time_from_text_or_llm(text, hint)
                if tfix:
                    s.appointment_time = tfix
                    return self._finalize_reschedule(), True
                if _utterance_has_date_signal(text):
                    dfix = self._date_from_text_or_llm(text, hint)
                    if dfix:
                        s.appointment_date = dfix
                        return f"New date set to {s.appointment_date}. What time?", False
                return (
                    "Say a time with a.m. or p.m., or tell me the new calendar date if you’re changing the day.",
                    False,
                )
            tm = self._time_from_text_or_llm(text, hint)
            if not tm and "morning" in _normalize(text):
                tm = "09:00"
            if not tm and "afternoon" in _normalize(text):
                tm = "14:00"
            if not tm and ("any" in _normalize(text) or "flexible" in _normalize(text)):
                tm = "10:00"
            if not tm:
                return "What time should I book on the new day?", False
            s.appointment_time = tm
            return self._finalize_reschedule(), True

        return "How can I help with your appointment?", True
