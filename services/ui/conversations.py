"""Previous conversations sidebar helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import get_conversation, list_conversations


def _label(conv: dict[str, Any]) -> str:
    preview = (conv.get("preview") or "Empty conversation").strip()
    updated = conv.get("update_ts") or ""
    if isinstance(updated, str) and "T" in updated:
        updated = updated.replace("T", " ")[:16]
    short_id = str(conv.get("conversation_id", ""))[:8]
    return f"{preview} · {updated} · {short_id}"


def render_conversation_sidebar(api_ok: bool) -> None:
    st.sidebar.header("Conversations")

    if st.sidebar.button("➕ New chat", use_container_width=True, disabled=not api_ok):
        st.session_state.ui_messages = []
        st.session_state.session_id = None
        st.rerun()

    if st.sidebar.button("🔄 Refresh list", use_container_width=True, disabled=not api_ok):
        st.rerun()

    if not api_ok:
        st.sidebar.caption("Connect the API to load previous chats.")
        return

    try:
        conversations = list_conversations(limit=50)
    except RuntimeError as exc:
        st.sidebar.error(str(exc))
        return

    if not conversations:
        st.sidebar.caption("No saved conversations yet.")
        return

    st.sidebar.caption("Select a previous conversation to continue it.")
    for conv in conversations:
        cid = str(conv["conversation_id"])
        is_active = st.session_state.session_id == cid
        button_label = f"{'▶ ' if is_active else ''}{_label(conv)}"
        if st.sidebar.button(
            button_label,
            key=f"conv_{cid}",
            use_container_width=True,
            disabled=is_active,
        ):
            try:
                detail = get_conversation(cid)
            except RuntimeError as exc:
                st.sidebar.error(str(exc))
                return
            st.session_state.session_id = cid
            st.session_state.ui_messages = detail.get("messages") or []
            st.rerun()
