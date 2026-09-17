"""
Multi-turn weather chat via the OpenAI Chat Completions API.
Uses OpenWeather tools for live current weather and multi-day forecasts.
Keeps the conversation open and sends full chat history on every turn.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError

load_dotenv()

SYSTEM_PROMPT = """You are a helpful weather assistant.

You have OpenWeather tools:
- `get_current_weather`: use for accurate current / live / right-now weather.
- `get_weather_forecast`: use for tomorrow, upcoming days, or any forecast request
  (OpenWeather free forecast is 5 days in 3-hour steps).

If the location is unclear, ask for clarification before calling a tool.
Base your answer on the tool result when you use a tool. Include temperature,
conditions, humidity, and wind when available, plus a short tip.
For forecast answers, summarize clearly by day or time when helpful.
For casual or hypothetical questions that do not need live data, you may answer
without a tool. Answer follow-up questions using the conversation history.
"""

EXIT_COMMANDS = {"quit", "exit", "q"}

LOCATION_PARAM = {
    "type": "string",
    "description": (
        "City name, optionally with state and country code, "
        "e.g. 'London', 'Austin,TX,US', or 'Tokyo,JP'."
    ),
}

UNITS_PARAM = {
    "type": "string",
    "enum": ["metric", "imperial", "standard"],
    "description": (
        "Temperature units: metric (°C), imperial (°F), "
        "or standard (Kelvin). Default metric."
    ),
}

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get accurate, live current weather for a city or location "
                "from the OpenWeather API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": LOCATION_PARAM,
                    "units": UNITS_PARAM,
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Get an accurate multi-day weather forecast (up to 5 days, "
                "3-hour steps) from the OpenWeather API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": LOCATION_PARAM,
                    "units": UNITS_PARAM,
                    "cnt": {
                        "type": "integer",
                        "description": (
                            "Number of 3-hour forecast slots to return (1-40). "
                            "Default 16 (~2 days). Use 40 for a full 5-day forecast."
                        ),
                        "minimum": 1,
                        "maximum": 40,
                    },
                },
                "required": ["location"],
            },
        },
    },
]

MAX_TOOL_ROUNDS = 5
TOOLS_REQUIRING_LOCATION = {"get_current_weather", "get_weather_forecast"}


def _unit_labels(units: str) -> dict[str, str]:
    return {
        "temperature": {"metric": "°C", "imperial": "°F", "standard": "K"}[units],
        "wind_speed": {"metric": "m/s", "imperial": "mph", "standard": "m/s"}[units],
    }


def _normalize_units(units: str | None) -> str:
    if units not in {"metric", "imperial", "standard"}:
        return "metric"
    return units


def _openweather_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET an OpenWeather endpoint and return parsed JSON or an error dict."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "OPENWEATHER_API_KEY is not set in .env"}

    query = urllib.parse.urlencode({**params, "appid": api_key})
    url = f"https://api.openweathermap.org/data/2.5/{path}?{query}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = {"message": body or str(exc)}
        return {
            "error": "OpenWeather request failed",
            "status": exc.code,
            "detail": detail,
        }
    except urllib.error.URLError as exc:
        return {"error": f"Could not reach OpenWeather: {exc.reason}"}
    except Exception as exc:
        return {"error": f"Unexpected OpenWeather failure: {exc}"}


def get_current_weather(location: str, units: str = "metric") -> dict[str, Any]:
    """Fetch current weather from OpenWeather for a location."""
    units = _normalize_units(units)
    data = _openweather_get("weather", {"q": location, "units": units})
    if "error" in data:
        return data

    weather = (data.get("weather") or [{}])[0]
    main = data.get("main") or {}
    wind = data.get("wind") or {}
    sys_info = data.get("sys") or {}

    return {
        "location": data.get("name") or location,
        "country": sys_info.get("country"),
        "description": weather.get("description"),
        "temperature": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "temp_min": main.get("temp_min"),
        "temp_max": main.get("temp_max"),
        "humidity_percent": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "units": _unit_labels(units),
        "source": "OpenWeather",
    }


def get_weather_forecast(
    location: str,
    units: str = "metric",
    cnt: int = 16,
) -> dict[str, Any]:
    """Fetch up to 5-day / 3-hour forecast from OpenWeather."""
    units = _normalize_units(units)
    try:
        cnt_int = int(cnt)
    except (TypeError, ValueError):
        cnt_int = 16
    cnt_int = max(1, min(cnt_int, 40))

    data = _openweather_get(
        "forecast",
        {"q": location, "units": units, "cnt": cnt_int},
    )
    if "error" in data:
        return data

    city = data.get("city") or {}
    forecasts: list[dict[str, Any]] = []
    for entry in data.get("list") or []:
        weather = (entry.get("weather") or [{}])[0]
        main = entry.get("main") or {}
        wind = entry.get("wind") or {}
        forecasts.append(
            {
                "time": entry.get("dt_txt"),
                "description": weather.get("description"),
                "temperature": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "temp_min": main.get("temp_min"),
                "temp_max": main.get("temp_max"),
                "humidity_percent": main.get("humidity"),
                "wind_speed": wind.get("speed"),
                "precipitation_probability": entry.get("pop"),
            }
        )

    return {
        "location": city.get("name") or location,
        "country": city.get("country"),
        "count": len(forecasts),
        "forecasts": forecasts,
        "units": _unit_labels(units),
        "source": "OpenWeather",
        "note": "Forecast slots are every 3 hours, up to 5 days.",
    }


TOOL_HANDLERS = {
    "get_current_weather": lambda args: get_current_weather(
        location=args.get("location", ""),
        units=args.get("units", "metric"),
    ),
    "get_weather_forecast": lambda args: get_weather_forecast(
        location=args.get("location", ""),
        units=args.get("units", "metric"),
        cnt=args.get("cnt", 16),
    ),
}


def _run_tool(name: str, arguments_json: str) -> str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        args = json.loads(arguments_json or "{}")
        if not isinstance(args, dict):
            return json.dumps({"error": "Tool arguments must be a JSON object"})
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON in tool arguments"})

    if name in TOOLS_REQUIRING_LOCATION and not str(args.get("location", "")).strip():
        return json.dumps({"error": "location is required"})

    result = handler(args)
    return json.dumps(result)


def chat_once(
    client: OpenAI,
    messages: list[dict],
    model: str = "gpt-4o-mini",
) -> str:
    """Send full history; resolve any tool calls; return the final assistant reply."""
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=messages,
            tools=WEATHER_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }
                for call in tool_calls
            ]
        messages.append(assistant_message)

        if not tool_calls:
            return (message.content or "").strip()

        for call in tool_calls:
            tool_output = _run_tool(call.function.name, call.function.arguments or "{}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_output,
                }
            )

    return "I hit the tool-call limit before finishing. Please try asking again."


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to ChatCompletionAPI/.env", file=sys.stderr)
        sys.exit(1)

    if not os.getenv("OPENWEATHER_API_KEY"):
        print(
            "Warning: OPENWEATHER_API_KEY not set. Live weather tool calls will fail.",
            file=sys.stderr,
        )

    client = OpenAI(api_key=api_key)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Weather chat started. Ask about any location.")
    print("Ask for current weather or a forecast to use OpenWeather tools.")
    print("Type 'quit', 'exit', or 'q' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            print("Please enter a question, or type 'quit' to exit.")
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            break

        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})

        try:
            reply = chat_once(client, messages)
        except AuthenticationError:
            print("Error: Invalid OpenAI API key. Check OPENAI_API_KEY in .env.", file=sys.stderr)
            del messages[checkpoint:]
            break
        except RateLimitError:
            print("Error: Rate limit exceeded. Please wait and try again.", file=sys.stderr)
            del messages[checkpoint:]
            continue
        except APIConnectionError:
            print("Error: Could not reach OpenAI. Check your network connection.", file=sys.stderr)
            del messages[checkpoint:]
            continue
        except APIError as exc:
            print(f"Error: OpenAI API request failed: {exc}", file=sys.stderr)
            del messages[checkpoint:]
            continue
        except Exception as exc:
            print(f"Error: Unexpected failure: {exc}", file=sys.stderr)
            del messages[checkpoint:]
            continue

        if not reply:
            print("Assistant: (empty response — try asking again)")
            del messages[checkpoint:]
            continue

        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
