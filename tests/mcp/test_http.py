"""The MCP server over real HTTP: bearer auth, health, and per-run trace logging (SPEC §5.3-5.4)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from ats.main import create_app
from ats.mcp_server import Runtime, create_mcp

from .conftest import make_settings

TOKEN = "test-mcp-token"


@contextmanager
def serve(runtime: Runtime) -> Iterator[str]:
    """Run the FastAPI app on a free local port in a background thread."""
    app = create_app(runtime, create_mcp(runtime))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=20)


def mcp_client(base: str, token: str | None = TOKEN, **headers: str) -> Client:
    return Client(StreamableHttpTransport(f"{base}/mcp/", headers=headers or None, auth=token))


@pytest.fixture
def secured(indexed, jobs_only_dir: Path, tmp_path: Path) -> Runtime:
    return Runtime(
        make_settings(
            data_dir=jobs_only_dir,
            chroma_dir=indexed[0],
            runs_dir=tmp_path / "runs",
            mcp_api_key=TOKEN,
        )
    )


async def test_mcp_requires_the_bearer_token(secured: Runtime) -> None:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    with serve(secured) as base:
        for headers in (
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": f"Basic {TOKEN}"},
            {"Authorization": TOKEN},
        ):
            response = httpx.post(f"{base}/mcp/", json=body, headers=headers)
            assert response.status_code == 401, headers
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.json() == {"error": "unauthorized"}
        with pytest.raises(Exception):  # noqa: B017 - the client wraps the 401 in its own error type
            async with mcp_client(base, token=None) as client:
                await client.list_tools()

        async with mcp_client(base) as client:
            assert len(await client.list_tools()) == 9
            jobs = await client.call_tool("list_jobs", {})
            assert jobs.data[0].job_id == "jd_analytics_mgr"

        # /api/health stays open and does not need the MCP token.
        health = httpx.get(f"{base}/api/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "vector_store_loaded": True}


async def test_mcp_is_open_when_no_key_is_set(jobs_only_dir: Path, tmp_path: Path) -> None:
    runtime = Runtime(make_settings(data_dir=jobs_only_dir, chroma_dir=tmp_path / "empty"))
    with serve(runtime) as base:
        async with mcp_client(base, token=None) as client:
            assert len(await client.list_tools()) == 9
        assert httpx.get(f"{base}/api/health").json() == {
            "status": "ok",
            "vector_store_loaded": False,
        }


async def test_tool_calls_are_traced_for_the_run_in_the_header(secured: Runtime) -> None:
    runs = secured.settings.runs_dir
    (runs / "run_abc").mkdir(parents=True)
    with serve(secured) as base:
        async with mcp_client(base, **{"X-Run-Id": "run_abc"}) as client:
            hits = await client.call_tool("search_resumes", {"query": "SQL", "k": 30})
            await client.call_tool("get_pool_stats", {"hits": hits.structured_content["result"]})
            failed = await client.call_tool("get_job", {"job_id": "nope"}, raise_on_error=False)
            assert failed.is_error
            [trace] = await client.read_resource("ats://runs/run_abc/trace")

        # No folder for the run, or a header that is not a valid run id: nothing is written.
        for run_id in ("run_missing", "../run_abc", "a/b", "x" * 80):
            async with mcp_client(base, **{"X-Run-Id": run_id}) as client:
                await client.call_tool("list_jobs", {})
        # And no header at all.
        async with mcp_client(base) as client:
            await client.call_tool("list_jobs", {})

    assert sorted(p.name for p in runs.iterdir()) == ["run_abc"]
    assert sorted(p.name for p in (runs / "run_abc").iterdir()) == ["trace.jsonl"]
    text = (runs / "run_abc" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.text == text
    events = [json.loads(line) for line in text.splitlines()]
    assert [e["tool"] for e in events] == ["search_resumes", "get_pool_stats", "get_job"]
    for event in events:
        assert event["run_id"] == "run_abc" and event["event"] == "tool_call"
        assert event["latency_ms"] >= 0 and event["ts"]

    search, stats, job = events
    assert search["input"] == {"query": "SQL", "k": 30}
    assert search["output"]["ok"] is True and search["output"]["result_chars"] > 1000
    assert search["output"]["redactions"] == 0
    # Long argument lists are cut down in the trace.
    assert len(stats["input"]["hits"]) == 21 and stats["input"]["hits"][-1] == "... (30 items)"
    assert job["input"] == {"job_id": "nope"}
    assert job["output"]["ok"] is False and job["output"]["error"] == "ToolError"
