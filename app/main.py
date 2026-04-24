from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.conversation import ConversationEngine
from app.llm import get_llm_runtime_status, llm_enabled, llm_model_label, llm_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sena.assistant")

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
LOG_FILE = ROOT / "logs" / "conversations.jsonl"

sessions: dict[str, ConversationEngine] = {}

# Keep API setup simple so local demo startup is predictable.
app = FastAPI(title="SENA Practice Voice Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(default="")


@app.on_event("startup")
def startup() -> None:
    # Creates log folder and prints startup summary.
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Assistant API ready; static=%s; llm=%s provider=%s",
        STATIC,
        llm_enabled(),
        llm_provider() or "off",
    )


@app.post("/api/session")
def create_session() -> dict[str, str]:
    # Starts a new conversation session and returns its id.
    sid = str(uuid.uuid4())
    sessions[sid] = ConversationEngine(log_path=LOG_FILE)
    logger.info("New session %s", sid)
    return {"session_id": sid}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    # Processes one caller message for an existing session.
    engine = sessions.get(req.session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Unknown session")
    reply, payload = engine.process_message(req.message)
    return {"reply": reply, **payload}


@app.post("/api/reset")
def reset_session(req: ChatRequest) -> dict[str, str]:
    # Resets dialog state for the given session id.
    engine = sessions.get(req.session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Unknown session")
    engine.reset()
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    # Basic health probe for local checks.
    return {"status": "ok"}


@app.get("/api/features")
def features() -> dict[str, Any]:
    # Returns LLM configuration/runtime fields used by the UI.
    prov = llm_provider()
    runtime = get_llm_runtime_status()
    return {
        "llm_configured": llm_enabled(),
        "llm_provider": prov,
        "llm_model": llm_model_label() or None,
        # Keeping this key for frontend compatibility.
        "model": llm_model_label() or None,
        "llm_runtime": runtime,
    }


@app.get("/")
def index() -> FileResponse:
    # Serves the single-page frontend.
    if not (STATIC / "index.html").exists():
        raise HTTPException(status_code=500, detail="UI bundle missing")
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
