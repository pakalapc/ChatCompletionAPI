"""Chat panel rendering helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

CHAT_PANEL_HEIGHT = 620

_USER_BUBBLE = (
    "max-width:78%;padding:0.7rem 0.9rem;border-radius:12px;line-height:1.45;"
    "white-space:pre-wrap;word-wrap:break-word;background:#ddf4ff;"
    "border:1px solid #218bff;color:#0a3069 !important;text-align:right;"
)
_BOT_BUBBLE = (
    "max-width:78%;padding:0.7rem 0.9rem;border-radius:12px;line-height:1.45;"
    "white-space:pre-wrap;word-wrap:break-word;background:#eef7f0;"
    "border:1px solid #1a7f37;color:#0d1117 !important;text-align:left;"
)
_USER_MARKER = (
    "flex-shrink:0;width:2.4rem;height:2.4rem;border-radius:50%;display:flex;"
    "align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;"
    "color:#ffffff !important;background:#0550ae;"
)
_BOT_MARKER = (
    "flex-shrink:0;width:2.4rem;height:2.4rem;border-radius:50%;display:flex;"
    "align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;"
    "color:#ffffff !important;background:#116329;"
)


def _escape(content: str) -> str:
    return (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def render_message_html(role: str, content: str) -> str:
    is_user = role == "user"
    marker = "YOU" if is_user else "BOT"
    label = "You" if is_user else "Weather Bot"
    label_color = "#0550ae" if is_user else "#116329"
    row_dir = "row-reverse" if is_user else "row"
    bubble = _USER_BUBBLE if is_user else _BOT_BUBBLE
    marker_style = _USER_MARKER if is_user else _BOT_MARKER
    safe = _escape(content)

    return f"""
    <div style="display:flex;flex-direction:{row_dir};align-items:flex-start;
                gap:0.6rem;margin:0.65rem 0;">
      <div style="{marker_style}">{marker}</div>
      <div style="{bubble}">
        <div style="font-size:0.72rem;font-weight:700;margin-bottom:0.25rem;
                    color:{label_color} !important;">{label}</div>
        <div style="color:inherit !important;">{safe}</div>
      </div>
    </div>
    """


def render_chat_panel(
    messages: list[dict[str, str]],
    *,
    stream_callback: Callable[[Any, Any], None] | None = None,
) -> None:
    """Render messages in a scrollable panel.

    If stream_callback is provided, it runs inside the same panel after existing
    messages, with (status_box, reply_box) placeholders for live streaming.
    """
    with st.container(height=CHAT_PANEL_HEIGHT, border=True):
        if not messages and stream_callback is None:
            st.caption("No messages yet. Say hello or ask about a city's weather.")
            return

        for msg in messages:
            st.markdown(
                render_message_html(msg["role"], msg["content"]),
                unsafe_allow_html=True,
            )

        if stream_callback is not None:
            status_box = st.empty()
            reply_box = st.empty()
            stream_callback(status_box, reply_box)
