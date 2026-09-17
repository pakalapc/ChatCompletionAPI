"""
Streamlit UI for the weather chat application.
Talks to the FastAPI endpoints defined in api.py.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ~60px per short message → ~10 messages visible before scroll
CHAT_PANEL_HEIGHT = 620
API_BASE_URL = os.getenv("WEATHER_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Weather App", page_icon="⛅", layout="centered")

st.markdown(
    """
    <style>
    .chat-panel {
        border: 1px solid #8c959f;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        background: #f6f8fa;
        color: #0d1117 !important;
    }
    .msg-row {
        display: flex;
        margin: 0.55rem 0;
        align-items: flex-start;
        gap: 0.6rem;
        color: #0d1117 !important;
    }
    .msg-row.user {
        flex-direction: row-reverse;
    }
    .marker {
        flex-shrink: 0;
        width: 2.4rem;
        height: 2.4rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .marker.user { background: #0550ae; }
    .marker.bot { background: #116329; }
    .bubble {
        max-width: 78%;
        padding: 0.65rem 0.85rem;
        border-radius: 12px;
        line-height: 1.45;
        white-space: pre-wrap;
        word-wrap: break-word;
        color: #0d1117 !important;
    }
    .bubble.user {
        background: #ddf4ff;
        border: 1px solid #218bff;
        text-align: right;
        color: #0a3069 !important;
    }
    .bubble.bot {
        background: #ffffff;
        border: 1px solid #8c959f;
        color: #0d1117 !important;
    }
    .role-label {
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        opacity: 1;
        color: #24292f !important;
    }
    .bubble.user .role-label {
        color: #0550ae !important;
    }
    .bubble.bot .role-label {
        color: #116329 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _api_request(
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    """Call a FastAPI endpoint and return parsed JSON."""
    url = f"{API_BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            message = parsed.get("detail", detail)
        except json.JSONDecodeError:
            message = detail or str(exc)
        raise RuntimeError(f"API {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach API at {API_BASE_URL}. "
            "Start it with: uvicorn api:app --reload"
        ) from exc


def _check_health() -> bool:
    try:
        result = _api_request("GET", "/health")
        return result.get("status") == "ok"
    except Exception:
        return False


def _init_state() -> None:
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None


def _render_message(role: str, content: str) -> str:
    is_user = role == "user"
    marker = "YOU" if is_user else "BOT"
    label = "You" if is_user else "Weather Bot"
    row_class = "user" if is_user else "bot"
    marker_class = "user" if is_user else "bot"
    bubble_class = "user" if is_user else "bot"
    safe = (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    return f"""
    <div class="msg-row {row_class}">
      <div class="marker {marker_class}">{marker}</div>
      <div class="bubble {bubble_class}">
        <div class="role-label">{label}</div>
        {safe}
      </div>
    </div>
    """


def _send(user_text: str) -> None:
    """POST /chat — keeps server-side history via session_id."""
    st.session_state.ui_messages.append({"role": "user", "content": user_text})

    payload: dict = {"message": user_text}
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id

    try:
        result = _api_request("POST", "/chat", payload)
    except RuntimeError as exc:
        st.session_state.ui_messages.pop()
        st.error(str(exc))
        return

    st.session_state.session_id = result.get("session_id")
    reply = (result.get("reply") or "").strip()
    if not reply:
        st.session_state.ui_messages.pop()
        st.error("Empty response from API.")
        return

    st.session_state.ui_messages.append({"role": "assistant", "content": reply})


def _clear_chat() -> None:
    """DELETE /chat/{session_id} then clear local UI state."""
    session_id = st.session_state.session_id
    if session_id:
        try:
            _api_request("DELETE", f"/chat/{session_id}")
        except RuntimeError as exc:
            st.warning(f"Could not reset API session: {exc}")

    st.session_state.ui_messages = []
    st.session_state.session_id = None


def main() -> None:
    _init_state()

    st.title("Welcome to Weather App")
    st.caption("Ask for current weather or a multi-day forecast.")

    api_ok = _check_health()
    if api_ok:
        st.success(f"API connected ({API_BASE_URL})", icon="✅")
    else:
        st.error(
            f"API not reachable at {API_BASE_URL}. "
            "Run `uvicorn api:app --reload` in another terminal."
        )

    with st.expander("Chat window", expanded=True):
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

        chat_box = st.container(height=CHAT_PANEL_HEIGHT, border=False)
        with chat_box:
            if not st.session_state.ui_messages:
                st.info("No messages yet. Say hello or ask about a city's weather.")
            else:
                # Chronological: start → end (oldest at top, newest at bottom)
                html_parts = [
                    _render_message(msg["role"], msg["content"])
                    for msg in st.session_state.ui_messages
                ]
                st.markdown("".join(html_parts), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            cols = st.columns([5, 1])
            with cols[0]:
                user_input = st.text_input(
                    "Message",
                    placeholder="e.g. What's the weather in Seattle?",
                    label_visibility="collapsed",
                    disabled=not api_ok,
                )
            with cols[1]:
                submitted = st.form_submit_button(
                    "Send",
                    use_container_width=True,
                    disabled=not api_ok,
                )

        if submitted and user_input.strip():
            with st.spinner("Thinking..."):
                _send(user_input.strip())
            st.rerun()

        if st.button("Clear chat"):
            _clear_chat()
            st.rerun()


if __name__ == "__main__":
    main()
