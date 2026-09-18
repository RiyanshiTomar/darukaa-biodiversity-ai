"""
main.py — FastAPI backend
---------------------------
Endpoints:
  POST /chat            -> free-text input, mandatory
  POST /chat/structured  -> free text + structured JSON (soil/climate/land fields), bonus requirement
  GET  /health           -> simple healthcheck

Run:
    uvicorn backend.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.append(str(Path(__file__).parent.parent))  # so `rag` package is importable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.reasoning_engine import BiodiversityReasoningEngine

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence Chatbot",
    description="RAG-grounded conversational system for evidence-backed biodiversity recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = BiodiversityReasoningEngine()


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    message: str


class StructuredChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    message: str
    structured_data: dict = Field(
        default_factory=dict,
        description="e.g. {'soil_organic_carbon_pct': 0.3, 'rainfall': 'low', 'crop': 'monoculture wheat', 'region': 'semi-arid'}",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    result = engine.handle_message(session_id=req.session_id, user_text=req.message)
    return {"session_id": req.session_id, **result}


@app.post("/chat/structured")
def chat_structured(req: StructuredChatRequest):
    result = engine.handle_message(
        session_id=req.session_id,
        user_text=req.message,
        structured=req.structured_data,
    )
    return {"session_id": req.session_id, **result}


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    state = engine.sessions.get(session_id)
    if not state:
        return {"history": []}
    return {"history": state.history, "known_slots": list(state.known_slots.keys())}
