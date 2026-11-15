"""Embedders, selected by ``EMBEDDING_PROVIDER``.

- ``fastembed``: a local ONNX model run through fastembed (default ``BAAI/bge-small-en-v1.5``).
  No API key and no torch. This is the project's embedding model.
- ``hash``: offline feature hashing over stemmed words. Deterministic, needs no key, and is meant
  for tests, CI, and local development. It matches on shared words only, so it behaves like a
  keyword method: do not use it for the embedding baseline or any reported result.
- ``openai``: any OpenAI-compatible ``/embeddings`` endpoint (``EMBEDDING_BASE_URL``).

Documents and queries are embedded through separate methods, because some models (BGE among them)
expect an instruction in front of a search query but not in front of the text being searched.

An embedder's ``name`` is stored with the index, so a server configured with a different embedder
than the one used at ingest is rejected instead of silently returning bad scores.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from ats.config import Settings
from ats.text import content_terms

PROVIDERS = ("fastembed", "hash", "openai")
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(RuntimeError):
    pass


class EmbeddingConfigError(EmbeddingError):
    pass


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per text to be indexed or compared as a whole document, in order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """The vector for a search query. Models that expect a query instruction add it here."""
        ...


def query_prefix_for(model: str) -> str:
    """BGE English models expect an instruction on short search queries, never on passages."""
    name = model.lower()
    return BGE_QUERY_PREFIX if name.startswith("baai/bge-") and "-en" in name else ""


class FastEmbedEmbedder:
    """Local ONNX embeddings through fastembed, L2-normalised.

    fastembed is imported, and the model loaded (and downloaded on first use), only when the
    first text is embedded. Hash mode and CI therefore never load it.
    """

    def __init__(self, model: str = DEFAULT_FASTEMBED_MODEL, cache_dir: Path | None = None) -> None:
        self.model = model
        self.name = f"fastembed:{model}"
        self.query_prefix = query_prefix_for(model)
        self._cache_dir = cache_dir
        self._lock = threading.Lock()
        self._impl: Any = None

    def _load(self) -> Any:
        with self._lock:
            if self._impl is None:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise EmbeddingConfigError(
                        "EMBEDDING_PROVIDER=fastembed needs the fastembed package; run `uv sync`"
                    ) from exc
                supported = {m["model"].lower() for m in TextEmbedding.list_supported_models()}
                if self.model.lower() not in supported:
                    raise EmbeddingConfigError(
                        f"fastembed does not support EMBEDDING_MODEL {self.model!r}"
                    )
                cache_dir = str(self._cache_dir) if self._cache_dir else None
                try:
                    self._impl = TextEmbedding(model_name=self.model, cache_dir=cache_dir)
                except Exception as exc:
                    raise EmbeddingError(
                        f"could not load embedding model {self.model!r} ({type(exc).__name__}). "
                        "It is downloaded on first use, so check the network connection, or "
                        "that EMBEDDING_CACHE_DIR holds the model when running offline."
                    ) from exc
            return self._impl

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for vec in self._load().embed(texts, batch_size=64):
            values = [float(x) for x in vec]
            norm = math.sqrt(sum(x * x for x in values))
            out.append([x / norm for x in values] if norm else values)
        if len(out) != len(texts):
            raise EmbeddingError(f"fastembed returned {len(out)} vectors for {len(texts)} texts")
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]


class HashEmbedder:
    """Signed feature hashing of stemmed terms with sublinear term frequency, L2-normalised."""

    def __init__(self, dim: int = 512) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        self.name = f"hash:bow-{dim}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for term, tf in Counter(content_terms(text)).items():
            # blake2b, not hash(): Python salts hash() per process, and vectors must be stable.
            h = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if h >> 63 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(tf))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # No content words (or they cancelled out): a fixed unit vector keeps cosine defined.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class OpenAICompatEmbedder:
    """Calls ``POST {base_url}/embeddings`` in batches, retrying rate limits and server errors."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        *,
        batch_size: int = 96,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.name = f"openai:{model}"
        host = httpx.URL(base_url).host
        if api_key.startswith("sk-ant-") and not host.endswith("anthropic.com"):
            # Refuse before any request is made: the key would otherwise be sent to another company.
            raise EmbeddingConfigError(
                f"EMBEDDING_API_KEY looks like an Anthropic (Claude) key, but embeddings go to "
                f"{host}. Anthropic keys cannot be used for embeddings; set EMBEDDING_API_KEY to "
                "a key from your embedding provider, or use EMBEDDING_PROVIDER=hash."
            )
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    def _post(self, batch: list[str]) -> list[list[float]]:
        last = "no attempt made"
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(min(2.0 ** (attempt - 1), 8.0))
            try:
                resp = self._client.post(
                    self._url, headers=self._headers, json={"model": self.model, "input": batch}
                )
            except httpx.TransportError as exc:
                last = f"network error: {type(exc).__name__}"
                continue
            if resp.status_code in (401, 403):
                # The response body is not echoed: providers may include part of the key in it.
                raise EmbeddingError(
                    f"embedding request rejected (HTTP {resp.status_code}); check EMBEDDING_API_KEY"
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
                continue
            if resp.status_code != 200:
                raise EmbeddingError(
                    f"embedding request failed: HTTP {resp.status_code}: {resp.text[:300]}"
                )
            try:
                rows = sorted(resp.json()["data"], key=lambda row: row["index"])
                vectors = [[float(x) for x in row["embedding"]] for row in rows]
            except (KeyError, TypeError, ValueError) as exc:
                raise EmbeddingError(f"unexpected embedding response shape: {exc!r}") from exc
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"embedding response had {len(vectors)} vectors for {len(batch)} inputs"
                )
            return vectors
        raise EmbeddingError(
            f"embedding request failed after {self._max_retries + 1} attempts: {last}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out += self._post(texts[i : i + self._batch_size])
        if len({len(v) for v in out}) > 1:
            raise EmbeddingError("embedding response contained vectors of different sizes")
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def embedder_from_settings(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.strip().lower()
    if not provider:
        raise EmbeddingConfigError(
            "EMBEDDING_PROVIDER is not set. Use 'fastembed' (local model, no API key) or 'hash' "
            "(offline word matching, for tests and local development only)."
        )
    if provider == "fastembed":
        model = settings.embedding_model.strip() or DEFAULT_FASTEMBED_MODEL
        return FastEmbedEmbedder(model, settings.embedding_cache_dir)
    if provider == "hash":
        return HashEmbedder()
    if provider == "openai":
        model = settings.embedding_model.strip()
        key = settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else ""
        missing = [n for n, v in (("EMBEDDING_MODEL", model), ("EMBEDDING_API_KEY", key)) if not v]
        if missing:
            raise EmbeddingConfigError(f"EMBEDDING_PROVIDER=openai needs {', '.join(missing)}")
        return OpenAICompatEmbedder(model, key, settings.embedding_base_url)
    raise EmbeddingConfigError(
        f"unknown EMBEDDING_PROVIDER {provider!r}; expected one of {list(PROVIDERS)}"
    )
