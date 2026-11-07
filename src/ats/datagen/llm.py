"""LLM access for the generator: a small protocol, env-selected providers, and a disk cache.

Uses ``DATAGEN_LLM_*`` env vars, deliberately separate from the app's ``LLM_*`` vars so the data
is written by a different model than the one that evaluates it.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

DEFAULT_CACHE_DIR = Path(".cache/datagen")


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str:
        """Return the raw text response for one system+user prompt."""
        ...


class OpenAIClient:
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, model: str, api_key: str) -> None:
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(block, "text", "") for block in resp.content)


PROVIDERS: dict[str, type] = {"openai": OpenAIClient, "anthropic": AnthropicClient}


def prompt_text(system: str, user: str) -> str:
    return f"{system}\n\n{user}"


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256((model + prompt).encode("utf-8")).hexdigest()


class CachedLLMClient:
    """Wraps a client; responses are stored in ``cache_dir/<sha256(model + prompt)>.json``."""

    def __init__(self, inner: LLMClient, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.inner = inner
        self.model = inner.model
        self.provider = getattr(inner, "provider", "llm")
        self.cache_dir = Path(cache_dir)

    def path_for(self, system: str, user: str) -> Path:
        return self.cache_dir / f"{cache_key(self.model, prompt_text(system, user))}.json"

    def complete(self, system: str, user: str) -> str:
        path = self.path_for(system, user)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        response = self.inner.complete(system, user)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps({"model": self.model, "response": response}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return response


class ConfigError(RuntimeError):
    pass


def client_from_env(
    env: Mapping[str, str] | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> LLMClient:
    """Build the provider client from ``DATAGEN_LLM_*``; wrap it in the disk cache unless
    ``cache_dir`` is None."""
    env = os.environ if env is None else env
    provider = env.get("DATAGEN_LLM_PROVIDER", "").strip().lower()
    model = env.get("DATAGEN_LLM_MODEL", "").strip()
    api_key = env.get("DATAGEN_LLM_API_KEY", "").strip()
    missing = [
        name
        for name, value in (
            ("DATAGEN_LLM_PROVIDER", provider),
            ("DATAGEN_LLM_MODEL", model),
            ("DATAGEN_LLM_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise ConfigError(f"missing env vars: {', '.join(missing)} (or use --offline)")
    if provider not in PROVIDERS:
        raise ConfigError(
            f"unknown DATAGEN_LLM_PROVIDER {provider!r}; expected one of {list(PROVIDERS)}"
        )
    try:
        client: LLMClient = PROVIDERS[provider](model=model, api_key=api_key)
    except ImportError as exc:
        raise ConfigError(
            f"provider {provider!r} needs its SDK: uv sync --extra datagen-llm ({exc})"
        ) from exc
    client.provider = provider  # type: ignore[attr-defined]
    if cache_dir is not None:
        return CachedLLMClient(client, cache_dir)
    return client
