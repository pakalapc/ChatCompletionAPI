"""
Streamlit UI for the weather chat application.
Talks to the FastAPI endpoints and can reopen previous conversations.
Streams bot replies from POST /chat/stream inside the chat panel.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from dotenv import load_dotenv

from api_client import API_BASE_URL, check_health, clear_session, stream_chat
from chat_panel import render_chat_panel, render_message_html
from conversations import render_conversation_sidebar
from styles import CHAT_CSS

load_dotenv()

st.set_page_config(page_title="Weather App", page_icon="⛅", layout="wide")
st.markdown(CHAT_CSS, unsafe_allow_html=True)


def _init_state() -> None:
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def _make_stream_callback(user_text: str):
    """Stream into placeholders that live inside the chat panel."""

    def _stream(status_box: Any, reply_box: Any) -> None:
        assembled = ""
        try:
            for event in stream_chat(user_text, st.session_state.session_id):
                etype = event.get("type")
                if etype == "session":
                    st.session_state.session_id = event.get("session_id")
                elif etype == "status":
                    status_box.caption(event.get("message") or "Working…")
                elif etype == "token":
                    assembled += event.get("text") or ""
                    reply_box.markdown(
                        render_message_html("assistant", assembled),
                        unsafe_allow_html=True,
                    )
                elif etype == "error":
                    status_box.empty()
                    reply_box.error(event.get("message") or "Streaming failed")
                    return
                elif etype == "done":
                    reply = (event.get("reply") or assembled).strip()
                    status_box.empty()
                    if not reply:
                        reply_box.error("Empty response from API.")
                        return
                    st.session_state.ui_messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    # Keep the final bubble visible in this slot until rerun.
                    reply_box.markdown(
                        render_message_html("assistant", reply),
                        unsafe_allow_html=True,
                    )
                    return
        except RuntimeError as exc:
            status_box.empty()
            reply_box.error(str(exc))

    return _stream


def _clear_chat() -> None:
    session_id = st.session_state.session_id
    if session_id:
        try:
            clear_session(session_id)
        except RuntimeError as exc:
            st.warning(f"Could not reset API session: {exc}")

    st.session_state.ui_messages = []
    st.session_state.session_id = None
    st.session_state.pending_prompt = None


def main() -> None:
    _init_state()

    api_ok = check_health()
    render_conversation_sidebar(api_ok)

    st.title("Welcome to Weather App")
    st.caption("Ask for current weather or a multi-day forecast.")

    if api_ok:
        st.success(f"API connected ({API_BASE_URL})", icon="✅")
    else:
        st.error(
            f"API not reachable at {API_BASE_URL}. "
            "Run `docker compose up --build`."
        )

    if st.session_state.session_id:
        st.caption(f"Active conversation: `{st.session_state.session_id}`")

    pending = st.session_state.pending_prompt
    stream_callback = _make_stream_callback(pending) if pending else None

    with st.expander("Chat window", expanded=True):
        # Stream inside the same panel as history so the reply doesn't jump.
        render_chat_panel(
            st.session_state.ui_messages,
            stream_callback=stream_callback,
        )

        if pending:
            # Streaming finished (or failed) inside the panel; clear and settle UI.
            st.session_state.pending_prompt = None
            st.rerun()

        with st.form("chat_form", clear_on_submit=True):
            cols = st.columns([5, 1])
            with cols[0]:
                user_input = st.text_input(
                    "Message",
                    placeholder="e.g. What's the weather in Seattle?",
                    label_visibility="collapsed",
                    disabled=not api_ok or st.session_state.pending_prompt is not None,
                )
            with cols[1]:
                submitted = st.form_submit_button(
                    "Send",
                    use_container_width=True,
                    disabled=not api_ok or st.session_state.pending_prompt is not None,
                )

        if submitted and user_input.strip():
            st.session_state.ui_messages.append(
                {"role": "user", "content": user_input.strip()}
            )
            st.session_state.pending_prompt = user_input.strip()
            st.rerun()

        if st.button("Clear chat (keep history in DB)"):
            _clear_chat()
            st.rerun()


if __name__ == "__main__":
    main()
