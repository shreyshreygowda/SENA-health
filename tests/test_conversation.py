from datetime import UTC, datetime

from app.conversation import (
    ConversationEngine,
    Intent,
    classify_intent,
    _extract_name,
    general_reply,
    match_canned_general,
    _parse_date_from_text,
    _parse_time_from_utterance,
)


def test_classify_intents():
    assert classify_intent("I need to book an appointment") == Intent.BOOK
    assert classify_intent("Please reschedule my visit") == Intent.RESCHEDULE
    assert classify_intent("What are your hours?") == Intent.GENERAL


def test_general_hours():
    assert "8 a.m." in general_reply("What time do you open?")


def test_booking_flow_with_iso_date():
    eng = ConversationEngine()
    r1, s1 = eng.process_message("I'd like to book an appointment")
    assert "name" in r1.lower()
    r2, s2 = eng.process_message("My name is Taylor Morgan")
    assert s2["patient_name"] == "Taylor Morgan"
    assert "date" in r2.lower()
    r3, s3 = eng.process_message("March 18, 2027")
    assert s3["appointment_date"] == "2027-03-18"
    assert "time" in r3.lower()
    r4, s4 = eng.process_message("10:30 am")
    assert "confirmed" in r4.lower()
    assert s4.get("done") is True


def test_correction_changes_date():
    eng = ConversationEngine()
    eng.process_message("Book an appointment")
    eng.process_message("Jordan Lee")
    eng.process_message("April 10, 2027")
    r, s = eng.process_message("Actually, April 12, 2027")
    assert s["appointment_date"] == "2027-04-12"
    assert "time" in r.lower()


def test_parse_polite_ordinals_and_weekday():
    eng = ConversationEngine()
    eng.process_message("Book an appointment")
    eng.process_message("Alex")
    r, s = eng.process_message("can I please do May 6th 2027")
    assert s["appointment_date"] == "2027-05-06", r
    eng2 = ConversationEngine()
    eng2.process_message("I need to book")
    eng2.process_message("Alex")
    r2, s2 = eng2.process_message("can I do Thursday May 17th")
    assert s2["appointment_date"].endswith("-05-17"), r2


def test_the_nth_of_may():
    base = datetime(2026, 4, 22, tzinfo=UTC)
    assert _parse_date_from_text("the 10th of may", base) == "2026-05-10"


def test_actually_tail_prefers_second_date():
    base = datetime(2026, 4, 22, tzinfo=UTC)
    d = _parse_date_from_text(
        "can I do it for next Wednesday actually can I do it for next Thursday",
        base,
    )
    assert d == "2026-04-23"


def test_long_booking_line_not_a_name():
    eng = ConversationEngine(enable_llm=False)
    eng.process_message("Book an appointment")
    r, s = eng.process_message(
        "can I book my appointment for next Wednesday wait actually can I do it for next Tuesday"
    )
    assert s.get("patient_name") is None
    assert "name" in r.lower()


def test_office_days_canned():
    r = match_canned_general("what days is your office")
    assert r and "Monday" in r


def test_appointment_types_canned():
    r = match_canned_general("what type of appointments do I have access to")
    assert r and "visit" in r.lower()


def test_doctor_question_canned():
    r = match_canned_general("what doctor am i seeing")
    assert r and ("chart" in r.lower() or "portal" in r.lower())


def test_tomorrow_not_extracted_as_name():
    assert _extract_name("hi can I book an appointment for tomorrow") is None


def test_book_for_tomorrow_then_asks_name():
    eng = ConversationEngine(enable_llm=False)
    r, s = eng.process_message("hi can I book an appointment for tomorrow")
    assert s.get("appointment_date")
    assert s.get("phase") == "collect_name"
    assert s.get("patient_name") is None
    assert "who" in r.lower()


def test_doctor_faq_during_booking():
    eng = ConversationEngine(enable_llm=False)
    eng.process_message("book an appointment")
    eng.process_message("Jane Doe")
    r, s = eng.process_message("what doctor am i seeing")
    assert "chart" in r.lower()
    assert s.get("phase") == "collect_date"


def test_time_correction_not_date():
    assert _parse_time_from_utterance("can i do 7 actually can i do 10") == "22:00"
    assert _parse_time_from_utterance("can i do 7 pm no actually can i do 10 pm") == "22:00"


def test_reschedule_logs_final():
    eng = ConversationEngine()
    eng.process_message("I need to reschedule")
    eng.process_message("Sam Rivera")
    eng.process_message("May 1, 2027")
    eng.process_message("May 8, 2027")
    r, s = eng.process_message("afternoon")
    assert "moved" in r.lower() or "May" in r
    assert s.get("done") is True
