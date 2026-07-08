"""Shared Groq chat-model factory.

One place to construct the LLM so the model name, retries, and key validation stay
consistent across the app. Import this instead of instantiating ChatGroq directly.
"""

from __future__ import annotations

from langchain_groq import ChatGroq

from app.config import settings


def build_chat_model(temperature: float = 0.0) -> ChatGroq:
    """Return a configured Groq chat model, validating the API key first.

    Temperature defaults to 0.0 so SQL generation is as deterministic as possible.
    """
    settings.require("groq_api_key")
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_retries=2,
    )
