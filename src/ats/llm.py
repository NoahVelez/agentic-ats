
"""Shared asynchronous LLM client for all ATS agents."""

from __future__ import annotations

from typing import Protocol

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ats.config import Settings, get_settings


def extract_json_text(response: str) -> str:
    """Return the JSON object text inside an LLM response.

    Claude has no JSON mode on this code path, so a reply may wrap the object
    in a Markdown fence or add a sentence before it. Returns the stripped
    response unchanged when no enclosing ``{...}`` is found.
    """
    cleaned = response.strip()

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        return cleaned

    return cleaned[start : end + 1]


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
        )

        return response.choices[0].message.content or ""


class AnthropicClient:
    """Async client for Anthropic Claude models.

    Token usage is accumulated across calls in ``tokens_in`` / ``tokens_out``
    so a screening run can report what it consumed.
    """

    # Current Claude models think adaptively by default and thinking tokens
    # count toward max_tokens, so leave enough room for the JSON answer.
    MAX_TOKENS = 16000

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)
        self.tokens_in = 0
        self.tokens_out = 0

    async def complete(self, system: str, user: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.tokens_in += usage.input_tokens or 0
            self.tokens_out += usage.output_tokens or 0

        if response.stop_reason == "max_tokens":
            raise ValueError(
                f"Claude response was truncated at max_tokens={self.MAX_TOKENS}."
            )

        if response.stop_reason == "refusal":
            raise ValueError("Claude declined to answer this request.")

        return "\n".join(
            block.text for block in response.content if block.type == "text"
        )


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

    if provider == "claude":
        return AnthropicClient(
            model=model,
            api_key=api_key,
        )

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
        "Supported providers: claude, groq, openai."
    )
