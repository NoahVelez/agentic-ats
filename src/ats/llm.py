
"""Shared asynchronous LLM client for all ATS agents."""

from __future__ import annotations

from typing import Protocol

from openai import AsyncOpenAI

from ats.config import Settings, get_settings


class LLMClient(Protocol):
    """Interface used by ATS agents."""

    async def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response."""
        ...


class OpenAICompatibleClient:
    """Async client for OpenAI and OpenAI-compatible providers such as Groq."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def complete(self, system: str, user: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        return response.choices[0].message.content or ""


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Create the configured application LLM client."""

    settings = settings or get_settings()

    provider = settings.llm_provider.strip().lower()
    model = settings.llm_model.strip()

    if not provider:
        raise ValueError("LLM_PROVIDER is not configured.")

    if not model:
        raise ValueError("LLM_MODEL is not configured.")

    if settings.llm_api_key is None:
        raise ValueError("LLM_API_KEY is not configured.")

    api_key = settings.llm_api_key.get_secret_value()

    if provider == "groq":
        return OpenAICompatibleClient(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    if provider == "openai":
        return OpenAICompatibleClient(
            model=model,
            api_key=api_key,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}. "
        "Supported providers: groq, openai."
    )
