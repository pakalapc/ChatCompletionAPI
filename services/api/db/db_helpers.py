"""Small helpers for conversation persistence used by the chat loop."""

from __future__ import annotations

from typing import Any

from db.persistence import add_step, to_jsonable


def persist_step(
    *,
    conversation_id: str | None,
    turn_id: str | None,
    step_type: str,
    input_data: Any = None,
    output_data: Any = None,
) -> None:
    """Persist one step when conversation_id and turn_id are both set; otherwise no-op."""
    if not (conversation_id and turn_id):
        return

    add_step(
        turn_id=turn_id,
        conversation_id=conversation_id,
        step_type=step_type,
        input_data=to_jsonable(input_data),
        output_data=to_jsonable(output_data),
    )
