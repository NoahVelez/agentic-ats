from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import httpx
import pytest

from ats.config import Settings
from ats.rag.embeddings import (
    BGE_QUERY_PREFIX,
    DEFAULT_FASTEMBED_MODEL,
    EmbeddingConfigError,
    EmbeddingError,
    FastEmbedEmbedder,
    HashEmbedder,
    OpenAICompatEmbedder,
    embedder_from_settings,
    query_prefix_for,
)

SECRET = "sk-test-SECRET-123"


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embedder_is_normalised_and_deterministic() -> None:
    emb = HashEmbedder()
    texts = ["Managed a team of 6 data analysts", "", "the and of"]
    vectors = emb.embed_documents(texts)
    assert vectors == HashEmbedder().embed_documents(texts)
    assert all(len(v) == 512 for v in vectors)
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, abs_tol=1e-9) for v in vectors)
    assert emb.name == "hash:bow-512" and HashEmbedder(256).name == "hash:bow-256"
    assert emb.embed_query(texts[0]) == vectors[0]


def test_hash_embedder_is_stable_across_processes() -> None:
    text = "Wrote complex SQL queries with window functions"
    code = (
        "import json; from ats.rag.embeddings import HashEmbedder; "
        f"print(json.dumps(HashEmbedder().embed_documents([{text!r}])[0]))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == HashEmbedder().embed_documents([text])[0]


def test_hash_embedder_scores_shared_words_higher() -> None:
    emb = HashEmbedder()
    query, close, far = emb.embed_documents(
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
    vectors = emb.embed_documents(["a", "bbb", "cc"])
    assert vectors == [[1.0, 0.0], [3.0, 1.0], [2.0, 0.0]]
    assert emb.name == "openai:small-model"
    assert emb.embed_query("dddd") == [4.0, 0.0]
    assert [str(r.url) for r in seen] == ["https://embed.example/v1/embeddings"] * 3
    assert all(r.headers["authorization"] == f"Bearer {SECRET}" for r in seen)
    assert json.loads(seen[0].content) == {"model": "small-model", "input": ["a", "bbb"]}


def test_openai_embedder_retries_rate_limits_then_succeeds() -> None:
    calls = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else ok_response(request)

    emb = OpenAICompatEmbedder("m", SECRET, client=make_client(handler), sleep=sleeps.append)
    assert emb.embed_documents(["abc"]) == [[3.0, 0.0]]
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


def test_openai_embedder_gives_up_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    emb = OpenAICompatEmbedder("m", SECRET, client=make_client(handler), sleep=lambda _: None)
    with pytest.raises(EmbeddingError, match="after 4 attempts: network error"):
        emb.embed_documents(["abc"])


def test_openai_embedder_auth_error_hides_key_and_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"Incorrect API key provided: {SECRET}"})

    emb = OpenAICompatEmbedder("m", SECRET, client=make_client(handler))
    with pytest.raises(EmbeddingError) as exc:
        emb.embed_documents(["abc"])
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
        emb.embed_documents(["abc"])


def test_anthropic_key_is_never_sent_to_another_embedding_host() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return ok_response(request)

    claude_key = "sk-ant-api03-not-a-real-key"
    with pytest.raises(EmbeddingConfigError, match="api.openai.com") as exc:
        OpenAICompatEmbedder("m", claude_key, client=make_client(handler))
    assert claude_key not in str(exc.value)
    with pytest.raises(EmbeddingConfigError, match="Anthropic"):
        embedder_from_settings(
            settings(embedding_provider="openai", embedding_model="m", embedding_api_key=claude_key)
        )
    assert sent == []
    # Only refused when the destination is not Anthropic itself.
    OpenAICompatEmbedder(
        "m", claude_key, "https://api.anthropic.com/v1", client=make_client(handler)
    )


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

    local = embedder_from_settings(settings(embedding_provider="fastembed"))
    assert isinstance(local, FastEmbedEmbedder)
    assert local.model == DEFAULT_FASTEMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert local.name == "fastembed:BAAI/bge-small-en-v1.5" and local._cache_dir is None
    custom = embedder_from_settings(
        settings(
            embedding_provider="fastembed",
            embedding_model="BAAI/bge-base-en-v1.5",
            embedding_cache_dir="models",
        )
    )
    assert custom.name == "fastembed:BAAI/bge-base-en-v1.5" and custom._cache_dir == Path("models")
    hosted = embedder_from_settings(
        settings(embedding_provider="openai", embedding_model="m1", embedding_api_key=SECRET)
    )
    assert isinstance(hosted, OpenAICompatEmbedder) and hosted.name == "openai:m1"


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding: records what it was given, returns raw vectors."""

    instances: list[FakeTextEmbedding] = []
    supported = (
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "broken/download",
    )

    @classmethod
    def list_supported_models(cls) -> list[dict[str, str]]:
        return [{"model": name} for name in cls.supported]

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        if model_name == "broken/download":
            raise ValueError("Could not load model broken/download from any source.")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.seen: list[str] = []
        FakeTextEmbedding.instances.append(self)

    def embed(self, documents, batch_size: int = 256):
        for text in documents:
            self.seen.append(text)
            yield [3.0, 4.0] if text.startswith(BGE_QUERY_PREFIX) else [0.0, 2.0]


@pytest.fixture
def fake_fastembed(monkeypatch):
    FakeTextEmbedding.instances = []
    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return FakeTextEmbedding


def test_fastembed_prefixes_queries_only_and_normalises(fake_fastembed, tmp_path) -> None:
    emb = FastEmbedEmbedder(cache_dir=tmp_path / "models")
    assert emb.name == "fastembed:BAAI/bge-small-en-v1.5"
    assert emb.query_prefix == "Represent this sentence for searching relevant passages: "
    # Nothing is loaded until the first text is embedded.
    assert fake_fastembed.instances == []

    docs = emb.embed_documents(["Led a 5-person analytics group", "Built SQL reports"])
    query = emb.embed_query("people management")
    [impl] = fake_fastembed.instances
    assert (impl.model_name, impl.cache_dir) == ("BAAI/bge-small-en-v1.5", str(tmp_path / "models"))
    assert impl.seen == [
        "Led a 5-person analytics group",
        "Built SQL reports",
        BGE_QUERY_PREFIX + "people management",
    ]
    # Raw vectors [0, 2] and [3, 4] come back with length 1.
    assert docs == [[0.0, 1.0], [0.0, 1.0]] and query == [0.6, 0.8]
    emb.embed_query("again")
    assert len(fake_fastembed.instances) == 1


def test_fastembed_query_prefix_only_for_bge_english_models(fake_fastembed) -> None:
    assert query_prefix_for("BAAI/bge-base-en-v1.5") == BGE_QUERY_PREFIX
    assert query_prefix_for("BAAI/bge-small-zh-v1.5") == ""
    other = FastEmbedEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    other.embed_query("people management")
    assert fake_fastembed.instances[0].seen == ["people management"]
    assert FastEmbedEmbedder(cache_dir=None).embed_documents(["x"]) == [[0.0, 1.0]]
    assert fake_fastembed.instances[1].cache_dir is None


def test_fastembed_configuration_errors(fake_fastembed, monkeypatch) -> None:
    with pytest.raises(
        EmbeddingConfigError, match="does not support EMBEDDING_MODEL 'not/a-model'"
    ):
        FastEmbedEmbedder("not/a-model").embed_query("x")
    # A supported model that cannot be loaded (no network, empty cache) is a different error.
    with pytest.raises(EmbeddingError, match="could not load.*EMBEDDING_CACHE_DIR") as exc:
        FastEmbedEmbedder("broken/download").embed_query("x")
    assert not isinstance(exc.value, EmbeddingConfigError)
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `import fastembed` fail
    with pytest.raises(EmbeddingConfigError, match="needs the fastembed package"):
        FastEmbedEmbedder().embed_documents(["x"])


def test_hash_mode_and_the_server_never_import_fastembed() -> None:
    code = (
        "import sys\n"
        "import ats.rag.embeddings, ats.rag.store, ats.rag.ingest, ats.mcp_server, ats.main\n"
        "from ats.rag.embeddings import HashEmbedder, FastEmbedEmbedder\n"
        "HashEmbedder().embed_query('x'); FastEmbedEmbedder()\n"
        "print('fastembed' in sys.modules)"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("EMBEDDING_")}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parent,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"
