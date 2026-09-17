"""HTTP client for the weather FastAPI backend."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_BASE_URL = os.getenv("WEATHER_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def api_request(
    method: str,
    path: str,
    payload: dict | None = None,
    query: dict[str, Any] | None = None,
) -> Any:
    """Call a FastAPI endpoint and return parsed JSON."""
    url = f"{API_BASE_URL}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

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
            "Start it with: docker compose up --build"
        ) from exc


def check_health() -> bool:
    try:
        result = api_request("GET", "/health")
        return result.get("status") == "ok"
    except Exception:
        return False


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    result = api_request("GET", "/conversations", query={"limit": limit})
    return result if isinstance(result, list) else []


def get_conversation(conversation_id: str) -> dict[str, Any]:
    result = api_request("GET", f"/conversations/{conversation_id}")
    return result if isinstance(result, dict) else {}


def send_chat(message: str, session_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    result = api_request("POST", "/chat", payload)
    return result if isinstance(result, dict) else {}


def stream_chat(message: str, session_id: str | None = None):
    """Yield parsed SSE events from POST /chat/stream."""
    payload: dict[str, Any] = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    url = f"{API_BASE_URL}/chat/stream"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw_buf = b""
            while True:
                chunk = response.read(256)
                if not chunk:
                    break
                raw_buf += chunk
                while b"\n" in raw_buf:
                    line_bytes, raw_buf = raw_buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        payload_text = line[5:].strip()
                        if not payload_text:
                            continue
                        try:
                            yield json.loads(payload_text)
                        except json.JSONDecodeError:
                            continue
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            message_text = parsed.get("detail", detail)
        except json.JSONDecodeError:
            message_text = detail or str(exc)
        raise RuntimeError(f"API {exc.code}: {message_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach API at {API_BASE_URL}. "
            "Start it with: docker compose up --build"
        ) from exc


def clear_session(session_id: str) -> None:
    api_request("DELETE", f"/chat/{session_id}")
