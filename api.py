"""
Simple FastAPI wrapper around Weather.chat_once.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from Weather import SYSTEM_PROMPT, chat_once

load_dotenv()

app = FastAPI(title="Weather Chat API", version="1.0.0")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to ChatCompletionAPI/.env")

client = OpenAI(api_key=api_key)

# session_id -> full chat message history (including system prompt)
sessions: dict[str, list[dict[str, Any]]] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


def _get_or_create_session(session_id: str | None) -> tuple[str, list[dict[str, Any]]]:
    sid = session_id or str(uuid4())
    if sid not in sessions:
        sessions[sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return sid, sessions[sid]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Append the user message and call chat_once once per request."""
    session_id, messages = _get_or_create_session(request.session_id)
    checkpoint = len(messages)
    messages.append({"role": "user", "content": request.message.strip()})

    try:
        reply = chat_once(client, messages)
    except AuthenticationError as exc:
        del messages[checkpoint:]
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key") from exc
    except RateLimitError as exc:
        del messages[checkpoint:]
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded") from exc
    except APIConnectionError as exc:
        del messages[checkpoint:]
        raise HTTPException(status_code=503, detail="Could not reach OpenAI") from exc
    except APIError as exc:
        del messages[checkpoint:]
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc
    except Exception as exc:
        del messages[checkpoint:]
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    if not reply:
        del messages[checkpoint:]
        raise HTTPException(status_code=502, detail="Empty response from model")

    return ChatResponse(session_id=session_id, reply=reply)


@app.delete("/chat/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    """Clear a conversation session."""
    sessions.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}
