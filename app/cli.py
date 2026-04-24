"""Text-only REPL for exercising the conversation engine without a browser."""

from __future__ import annotations

import logging
from pathlib import Path

from app.conversation import ConversationEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    # Handy local loop for testing flows without opening the browser UI.
    log = Path(__file__).resolve().parent.parent / "logs" / "cli.jsonl"
    engine = ConversationEngine(log_path=log)
    print("SENA practice assistant (text mode). Commands: /reset /quit")
    print("Try: I need to book an appointment for Jordan Smith next Friday morning\n")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            print("Assistant: I didn't catch that — could you repeat?")
            continue
        if line == "/quit":
            break
        if line == "/reset":
            engine.reset()
            print("Assistant: Session cleared. How can I help?")
            continue
        reply, state = engine.process_message(line)
        print(f"Assistant: {reply}")
        if state.get("done"):
            engine.reset()
            print("(Session reset for the next caller.)\n")


if __name__ == "__main__":
    main()
