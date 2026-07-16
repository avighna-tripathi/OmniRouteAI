"""Small async adapter around Google's official Gemini Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st
from google import genai
from google.genai import types


def get_gemini_api_key() -> str:
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key or key.startswith("your-"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your Google AI Studio key to .streamlit/secrets.toml."
        )
    return key


def _message_text(messages: list[Any]) -> str:
    """Convert LangChain-style messages to one Gemini prompt."""
    sections: list[str] = []
    for message in messages:
        role = getattr(message, "type", "message")
        content = getattr(message, "content", message)
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            content = "\n".join(text_parts)
        sections.append(f"[{role.upper()}]\n{content}")
    return "\n\n".join(sections)


@dataclass
class GeminiResponse:
    content: str


class GeminiModel:
    """Compatibility wrapper exposing the `ainvoke` shape used by the agents."""

    def __init__(self, model_id: str, temperature: float = 0.1):
        self.model_id = model_id
        self.temperature = temperature
        self.client = genai.Client(api_key=get_gemini_api_key())

    async def ainvoke(self, messages: list[Any]) -> GeminiResponse:
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=_message_text(messages),
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=int(st.secrets.get("GEMINI_MAX_OUTPUT_TOKENS", 4096)),
            ),
        )
        return GeminiResponse(content=response.text or "")

    async def generate_content(self, contents: list[Any]) -> GeminiResponse:
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=int(st.secrets.get("GEMINI_MAX_OUTPUT_TOKENS", 4096)),
            ),
        )
        return GeminiResponse(content=response.text or "")
