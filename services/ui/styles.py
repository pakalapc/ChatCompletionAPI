"""Shared Streamlit CSS for the weather chat UI."""

CHAT_CSS = """
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
.empty-chat {
    color: #57606a !important;
    font-size: 0.95rem;
    padding: 0.5rem 0.25rem;
}
</style>
"""
