from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import httpx
import pytest

from ats.config import Settings
from ats.rag.embeddings import (
    EmbeddingConfigError,
    EmbeddingError,
    HashEmbedder,
    OpenAICompatEmbedder,
    embedder_from_settings,
)

SECRET = "sk-test-SECRET-123"


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embedder_is_normalised_and_deterministic() -> None:
    emb = HashEmbedder()
    texts = ["Managed a team of 6 data analysts", "", "the and of"]
    vectors = emb.embed(texts)
    assert vectors == HashEmbedder().embed(texts)
    assert all(len(v) == 512 for v in vectors)
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, abs_tol=1e-9) for v in vectors)
    assert emb.name == "hash:bow-512" and HashEmbedder(256).name == "hash:bow-256"


def test_hash_embedder_is_stable_across_processes() -> None:
    text = "Wrote complex SQL queries with window functions"
    code = (
        "import json; from ats.rag.embeddings import HashEmbedder; "
        f"print(json.dumps(HashEmbedder().embed([{text!r}])[0]))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == HashEmbedder().embed([text])[0]


def test_hash_embedder_scores_shared_words_higher() -> None:
    emb = HashEmbedder()
    query, close, far = emb.embed(
        [
            "people management of analysts",
            "Provided people management for 8 analysts",
            "Prepared 300 covers per night on the saute station",
        ]
    )
    assert cosine(query, close) > cosine(query, far)


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_response(request: httpx.Request) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    # Returned out of order on purpose: the client must sort by "index".
    data = [{"index": i, "embedding": [float(len(t)), float(i)]} for i, t in enumerate(inputs)]
    return httpx.Response(200, json={"data": list(reversed(data))})


def test_openai_embedder_batches_orders_and_authenticates() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok_response(request)

    emb = OpenAICompatEmbedder(
        "small-model",
        SECRET,
        "https://embed.example/v1/",
        batch_size=2,
        client=make_client(handler),
    )
    vectors = emb.embed(["a", "bbb", "cc"])
    assert vectors == [[1.0, 0.0], [3.0, 1.0], [2.0, 0.0]]
    assert emb.name == "openai:small-model"
    assert [str(r.url) for r in seen] == ["https://embed.example/v1/embeddings"] * 2
    assert all(r.headers["authorization"] == f"Bearer {SECRET}" for r in seen)
    assert json.loads(seen[0].content) == {"model": "small-model", "input": ["a", "bbb"]}


def test_openai_embedder_retries_rate_limits_then_succeeds() -> None:
    calls = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else ok_response(request)

    emb = OpenAICompatEmbedder("m", SECRET, client=make_client(handler), sleep=sleeps.append)
    assert emb.embed(["abc"]) == [[3.0, 0.0]]
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


def test_openai_embedder_gives_up_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    emb = OpenAICompatEmbedder("m", SECRET, client=make_client(handler), sleep=lambda _: None)
    with pytest.raises(EmbeddingError, match="after 4 attempts: network error"):
        emb.embed(["abc"])


def test_openai_embedder_auth_error_hides_key_and_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"Incorrect API key provided: {SECRET}"})

    emb = OpenAICompatEmbedder("m", SECRET, client=make_client(handler))
    with pytest.raises(EmbeddingError) as exc:
        emb.embed(["abc"])
    assert "EMBEDDING_API_KEY" in str(exc.value) and SECRET not in str(exc.value)


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(400, text="bad model"), "HTTP 400: bad model"),
        (httpx.Response(200, json={"data": []}), "0 vectors for 1 inputs"),
        (httpx.Response(200, json={"oops": 1}), "unexpected embedding response shape"),
    ],
)
def test_openai_embedder_rejects_bad_responses(response: httpx.Response, match: str) -> None:
    emb = OpenAICompatEmbedder("m", SECRET, client=make_client(lambda request: response))
    with pytest.raises(EmbeddingError, match=match):
        emb.embed(["abc"])


def settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_embedder_from_settings(monkeypatch) -> None:
    for var in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(EmbeddingConfigError, match="EMBEDDING_PROVIDER is not set"):
        embedder_from_settings(settings())
    with pytest.raises(EmbeddingConfigError, match="unknown EMBEDDING_PROVIDER"):
        embedder_from_settings(settings(embedding_provider="nope"))
    with pytest.raises(EmbeddingConfigError, match="EMBEDDING_MODEL, EMBEDDING_API_KEY"):
        embedder_from_settings(settings(embedding_provider="openai"))
    assert isinstance(embedder_from_settings(settings(embedding_provider="HASH")), HashEmbedder)
    hosted = embedder_from_settings(
        settings(embedding_provider="openai", embedding_model="m1", embedding_api_key=SECRET)
    )
    assert isinstance(hosted, OpenAICompatEmbedder) and hosted.name == "openai:m1"
