"""
FastAPI weather chat backend.
Persists conversations / turns / steps in Postgres.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Iterator
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from Weather import SYSTEM_PROMPT, chat_once, chat_stream
from db import (
    conversation_exists,
    create_conversation,
    create_turn,
    get_conversation_ui_messages,
    init_schema,
    list_conversations,
    rebuild_llm_messages,
    touch_conversation,
)

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to .env")

client = OpenAI(api_key=api_key)

# conversation_id -> in-memory message history for active sessions
sessions: dict[str, list[dict[str, Any]]] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_schema()
        logger.info("Database schema ready")
    except Exception:
        logger.exception("Failed to initialize database schema")
        raise
    yield


app = FastAPI(title="Weather Chat API", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    user_id: str = "anonymous"


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ConversationSummary(BaseModel):
    conversation_id: str
    user_id: str
    creation_ts: datetime
    update_ts: datetime
    preview: str | None = None


class ConversationDetail(BaseModel):
    conversation_id: str
    messages: list[dict[str, str]]


def _get_or_create_session(
    session_id: str | None,
    user_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    """session_id is the conversation_id."""
    conversation_id = session_id or str(uuid4())

    if conversation_id not in sessions:
        if conversation_exists(conversation_id):
            sessions[conversation_id] = rebuild_llm_messages(
                conversation_id, SYSTEM_PROMPT
            )
        else:
            create_conversation(conversation_id, user_id)
            sessions[conversation_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    return conversation_id, sessions[conversation_id]


def _preview_for(conversation_id: str) -> str | None:
    messages = get_conversation_ui_messages(conversation_id)
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            text = msg["content"].strip().replace("\n", " ")
            return text if len(text) <= 80 else text[:77] + "..."
    return None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/conversations", response_model=list[ConversationSummary])
def get_conversations(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ConversationSummary]:
    rows = list_conversations(user_id=user_id, limit=limit)
    return [
        ConversationSummary(
            conversation_id=str(row["conversation_id"]),
            user_id=row["user_id"],
            creation_ts=row["creation_ts"],
            update_ts=row["update_ts"],
            preview=_preview_for(str(row["conversation_id"])),
        )
        for row in rows
    ]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> ConversationDetail:
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Warm in-memory session so follow-up /chat continues correctly
    sessions[conversation_id] = rebuild_llm_messages(conversation_id, SYSTEM_PROMPT)
    return ConversationDetail(
        conversation_id=conversation_id,
        messages=get_conversation_ui_messages(conversation_id),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """One user question → one turn (with intermediate steps persisted)."""
    conversation_id, messages = _get_or_create_session(
        request.session_id,
        request.user_id,
    )
    checkpoint = len(messages)
    messages.append({"role": "user", "content": request.message.strip()})

    turn_id = create_turn(conversation_id)

    try:
        reply = chat_once(
            client,
            messages,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
    except AuthenticationError as exc:
        del messages[checkpoint:]
        touch_conversation(conversation_id)
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key") from exc
    except RateLimitError as exc:
        del messages[checkpoint:]
        touch_conversation(conversation_id)
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded") from exc
    except APIConnectionError as exc:
        del messages[checkpoint:]
        touch_conversation(conversation_id)
        raise HTTPException(status_code=503, detail="Could not reach OpenAI") from exc
    except APIError as exc:
        del messages[checkpoint:]
        touch_conversation(conversation_id)
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc
    except Exception as exc:
        del messages[checkpoint:]
        touch_conversation(conversation_id)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    if not reply:
        del messages[checkpoint:]
        touch_conversation(conversation_id)
        raise HTTPException(status_code=502, detail="Empty response from model")

    return ChatResponse(session_id=conversation_id, reply=reply)


@app.post("/chat/stream")
def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """Stream the final assistant reply as SSE after resolving any tool calls."""
    conversation_id, messages = _get_or_create_session(
        request.session_id,
        request.user_id,
    )
    checkpoint = len(messages)
    messages.append({"role": "user", "content": request.message.strip()})
    turn_id = create_turn(conversation_id)

    def event_generator() -> Iterator[str]:
        try:
            yield _sse({"type": "session", "session_id": conversation_id})
            for event in chat_stream(
                client,
                messages,
                conversation_id=conversation_id,
                turn_id=turn_id,
            ):
                if event.get("type") == "error":
                    del messages[checkpoint:]
                    touch_conversation(conversation_id)
                yield _sse(event)
        except Exception as exc:
            del messages[checkpoint:]
            touch_conversation(conversation_id)
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.delete("/chat/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    """End the in-memory conversation; DB history is retained."""
    sessions.pop(session_id, None)
    if conversation_exists(session_id):
        touch_conversation(session_id)
    return {"status": "reset", "session_id": session_id}
