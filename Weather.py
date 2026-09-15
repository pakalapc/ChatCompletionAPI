"""
Multi-turn weather chat via the OpenAI Chat Completions API.
Keeps the conversation open and sends full chat history on every turn.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError

load_dotenv()

SYSTEM_PROMPT = """You are a helpful weather assistant.
Given a location, provide a clear, concise weather summary.
Include temperature (approx), conditions, humidity/wind if useful, and a short tip.
If the location is unclear, say so and ask for clarification.
Answer follow-up questions using the conversation history.
"""

EXIT_COMMANDS = {"quit", "exit", "q"}


def chat_once(client: OpenAI, messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Send full message history to Chat Completions and return the assistant reply."""
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to ChatCompletionAPI/.env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Weather chat started. Ask about any location.")
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

        messages.append({"role": "user", "content": user_input})

        try:
            reply = chat_once(client, messages)
        except AuthenticationError:
            print("Error: Invalid OpenAI API key. Check OPENAI_API_KEY in .env.", file=sys.stderr)
            messages.pop()
            break
        except RateLimitError:
            print("Error: Rate limit exceeded. Please wait and try again.", file=sys.stderr)
            messages.pop()
            continue
        except APIConnectionError:
            print("Error: Could not reach OpenAI. Check your network connection.", file=sys.stderr)
            messages.pop()
            continue
        except APIError as exc:
            print(f"Error: OpenAI API request failed: {exc}", file=sys.stderr)
            messages.pop()
            continue
        except Exception as exc:
            print(f"Error: Unexpected failure: {exc}", file=sys.stderr)
            messages.pop()
            continue

        if not reply:
            print("Assistant: (empty response — try asking again)")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
