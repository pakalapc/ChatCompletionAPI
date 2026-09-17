"""Postgres persistence for conversations, turns, and steps."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# Approximate gpt-4o-mini list prices (USD per 1M tokens). Used only to fill `cost`.
INPUT_USD_PER_1M = Decimal("0.15")
OUTPUT_USD_PER_1M = Decimal("0.60")

STEP_LLM_CALL = "llm_call"
STEP_TOOL_CALL = "tool_call"
STEP_FINAL_RENDER = "final_render"


def database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://weather:weather@localhost:5432/weather_chat",
    )


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(database_url(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    sql = schema_path.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with get_conn() as conn:
        for statement in statements:
            conn.execute(statement)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_conversation(conversation_id: str, user_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversations (conversation_id, user_id, creation_ts, update_ts)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (conversation_id) DO NOTHING
            """,
            (conversation_id, user_id, _now(), _now()),
        )


def touch_conversation(conversation_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET update_ts = %s
            WHERE conversation_id = %s
            """,
            (_now(), conversation_id),
        )


def create_turn(conversation_id: str) -> str:
    turn_id = str(uuid4())
    now = _now()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO turns (
                turn_id, conversation_id, creation_ts, update_ts,
                input_token_ct, output_token_ct, cost
            )
            VALUES (%s, %s, %s, %s, 0, 0, 0)
            """,
            (turn_id, conversation_id, now, now),
        )
    return turn_id


def update_turn_usage(
    turn_id: str,
    input_token_ct: int,
    output_token_ct: int,
) -> Decimal:
    cost = (
        Decimal(input_token_ct) * INPUT_USD_PER_1M
        + Decimal(output_token_ct) * OUTPUT_USD_PER_1M
    ) / Decimal("1000000")
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE turns
            SET input_token_ct = %s,
                output_token_ct = %s,
                cost = %s,
                update_ts = %s
            WHERE turn_id = %s
            """,
            (input_token_ct, output_token_ct, cost, _now(), turn_id),
        )
    return cost


def add_step(
    *,
    turn_id: str,
    conversation_id: str,
    step_type: str,
    input_data: Any = None,
    output_data: Any = None,
) -> str:
    step_id = str(uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO steps (
                step_id, turn_id, conversation_id, ts, step_type, input, output
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                step_id,
                turn_id,
                conversation_id,
                _now(),
                step_type,
                Jsonb(input_data) if input_data is not None else None,
                Jsonb(output_data) if output_data is not None else None,
            ),
        )
    return step_id


def conversation_exists(conversation_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
    return row is not None


def list_conversations(
    *,
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent conversations, newest update first."""
    limit = max(1, min(int(limit), 200))
    with get_conn() as conn:
        if user_id:
            rows = conn.execute(
                """
                SELECT conversation_id, user_id, creation_ts, update_ts
                FROM conversations
                WHERE user_id = %s
                ORDER BY update_ts DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT conversation_id, user_id, creation_ts, update_ts
                FROM conversations
                ORDER BY update_ts DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
    return [dict(row) for row in rows]


def _steps_for_conversation(conversation_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT s.step_id, s.turn_id, s.conversation_id, s.ts, s.step_type,
                   s.input, s.output, t.creation_ts AS turn_creation_ts
            FROM steps s
            JOIN turns t ON t.turn_id = s.turn_id
            WHERE s.conversation_id = %s
            ORDER BY t.creation_ts ASC, s.ts ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation_ui_messages(conversation_id: str) -> list[dict[str, str]]:
    """Rebuild user/assistant chat bubbles from persisted steps."""
    steps = _steps_for_conversation(conversation_id)
    ui_messages: list[dict[str, str]] = []

    current_turn: str | None = None
    user_for_turn: str | None = None
    reply_for_turn: str | None = None

    def _flush() -> None:
        nonlocal user_for_turn, reply_for_turn
        if user_for_turn:
            ui_messages.append({"role": "user", "content": user_for_turn})
        if reply_for_turn:
            ui_messages.append({"role": "assistant", "content": reply_for_turn})
        user_for_turn = None
        reply_for_turn = None

    for step in steps:
        turn_id = str(step["turn_id"])
        if current_turn is None:
            current_turn = turn_id
        elif turn_id != current_turn:
            _flush()
            current_turn = turn_id

        step_type = step["step_type"]
        input_data = step.get("input") or {}
        output_data = step.get("output") or {}

        if step_type == STEP_LLM_CALL and user_for_turn is None:
            messages = input_data.get("messages") if isinstance(input_data, dict) else None
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            user_for_turn = content
                        break

        if step_type == STEP_FINAL_RENDER and isinstance(output_data, dict):
            reply = output_data.get("reply")
            if isinstance(reply, str) and reply.strip():
                reply_for_turn = reply

    _flush()
    return ui_messages


def rebuild_llm_messages(conversation_id: str, system_prompt: str) -> list[dict[str, Any]]:
    """Rebuild OpenAI-style message history so a conversation can continue."""
    steps = _steps_for_conversation(conversation_id)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    current_turn: str | None = None
    user_added_for_turn = False

    for step in steps:
        turn_id = str(step["turn_id"])
        if turn_id != current_turn:
            current_turn = turn_id
            user_added_for_turn = False

        step_type = step["step_type"]
        input_data = step.get("input") or {}
        output_data = step.get("output") or {}

        if step_type == STEP_LLM_CALL:
            if not user_added_for_turn and isinstance(input_data, dict):
                prior = input_data.get("messages")
                if isinstance(prior, list):
                    for msg in reversed(prior):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            messages.append(
                                {"role": "user", "content": msg.get("content") or ""}
                            )
                            user_added_for_turn = True
                            break

            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": output_data.get("content") if isinstance(output_data, dict) else None,
            }
            tool_calls = (
                output_data.get("tool_calls") if isinstance(output_data, dict) else None
            )
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)

        elif step_type == STEP_TOOL_CALL:
            tool_call_id = (
                input_data.get("tool_call_id") if isinstance(input_data, dict) else None
            )
            result = output_data.get("result") if isinstance(output_data, dict) else None
            if not isinstance(result, str):
                result = json.dumps(result) if result is not None else ""
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }
            )

    return messages


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion for JSONB payloads."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)
