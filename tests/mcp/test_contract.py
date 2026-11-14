"""MCP contract tests through the in-memory client (SPEC §5, §15)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from ats.mcp_server import Runtime, create_mcp, pool_stats, scrub_result
from ats.models import ARCHETYPES, Chunk, SearchHit
from ats.rag.embeddings import HashEmbedder
from ats.rag.redaction import PII_FIELDS, REDACTED
from ats.rag.store import ResumeStore
from ats.text import PII_PATTERNS

from ..conftest import JOB_ID
from .conftest import make_settings

EXPECTED_TOOLS = {
    "get_job": ["job_id"],
    "list_jobs": [],
    "search_resumes": ["query", "k", "section"],
    "get_pool_stats": ["hits"],
    "get_candidate_chunks": ["candidate_id", "sections"],
    "search_candidate": ["candidate_id", "query", "k"],
    "verify_quote": ["candidate_id", "chunk_id", "quote"],
    "keyword_baseline": ["job_id", "top_n"],
    "embedding_baseline": ["job_id", "top_n"],
}
EXPECTED_DEFAULTS = {
    ("search_resumes", "k"): 20,
    ("search_candidate", "k"): 5,
    ("keyword_baseline", "top_n"): 20,
    ("embedding_baseline", "top_n"): 20,
}


async def call(client: Client, name: str, **args):
    return (await client.call_tool(name, args)).data


def keys_of(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_of(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_of(v)}
    return set()


async def every_tool_output(client: Client, resumes) -> tuple[list, str]:
    """Call every tool across the whole pool. Returns (structured results, all text joined)."""
    structured: list = []
    texts: list[str] = []

    async def record(name: str, **args):
        result = await client.call_tool(name, args)
        structured.append(result.structured_content)
        texts.extend(block.text for block in result.content)
        return result

    await record("list_jobs")
    await record("get_job", job_id=JOB_ID)
    await record("keyword_baseline", job_id=JOB_ID, top_n=1000)
    await record("embedding_baseline", job_id=JOB_ID, top_n=1000)
    queries = ["people management", "SQL", "dashboards", "email phone address name"]
    # Searching for a person must not surface them either.
    queries += [r.pii.name for r in resumes[:5]] + [resumes[0].pii.email]
    for query in queries:
        hits = await record("search_resumes", query=query, k=100)
        await record("get_pool_stats", hits=hits.structured_content["result"])
    for resume in resumes:
        chunks = await record("get_candidate_chunks", candidate_id=resume.candidate_id)
        await record("search_candidate", candidate_id=resume.candidate_id, query=resume.pii.name)
        first = chunks.structured_content["result"][0]
        await record(
            "verify_quote",
            candidate_id=resume.candidate_id,
            chunk_id=first["chunk_id"],
            quote=first["text"][:20],
        )
    return structured, "\n".join(texts) + "\n" + json.dumps(structured)


async def test_tool_surface_matches_the_spec(client: Client) -> None:
    tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) == set(EXPECTED_TOOLS)
    for name, params in EXPECTED_TOOLS.items():
        tool = tools[name]
        assert tool.description and tool.inputSchema["type"] == "object", name
        assert list(tool.inputSchema.get("properties", {})) == params, name
        assert tool.outputSchema, name
        assert tool.annotations.readOnlyHint is True, name
    for (name, param), default in EXPECTED_DEFAULTS.items():
        assert tools[name].inputSchema["properties"][param]["default"] == default
    assert tools["verify_quote"].inputSchema["required"] == ["candidate_id", "chunk_id", "quote"]


async def test_resources_and_prompts(client: Client) -> None:
    templates = {t.uriTemplate for t in await client.list_resource_templates()}
    assert templates == {"ats://jobs/{job_id}", "ats://runs/{run_id}/trace"}
    [content] = await client.read_resource(f"ats://jobs/{JOB_ID}")
    assert content.text.startswith("# Analytics Manager") and content.mimeType == "text/markdown"
    assert content.text == (await call(client, "get_job", job_id=JOB_ID)).text

    prompts = {p.name for p in await client.list_prompts()}
    assert prompts == {"requirements_system", "evaluator_system", "verifier_system"}
    for name in prompts:
        result = await client.get_prompt(name)
        assert len(result.messages[0].content.text) > 200

    for uri in ("ats://jobs/nope", "ats://runs/unknown-run/trace", "ats://runs/..%2F..%2Fx/trace"):
        with pytest.raises(Exception, match="unknown job_id|no trace for that run id"):
            await client.read_resource(uri)


async def test_no_tool_returns_pii(client: Client, resumes) -> None:
    structured, blob = await every_tool_output(client, resumes)
    assert not keys_of(structured) & PII_FIELDS
    lowered = blob.lower()
    for resume in resumes:
        pii = resume.pii
        street = pii.address.split(",")[0]
        for value in (pii.name, pii.email, pii.phone, street, pii.dob):
            assert value.lower() not in lowered, (resume.candidate_id, value)
    for kind, pattern in PII_PATTERNS:
        assert not pattern.search(blob), kind
    assert set(re.findall(r"cand_\d{4}", blob)) == {r.candidate_id for r in resumes}


async def test_answer_key_is_out_of_reach(client: Client, runtime: Runtime, resumes) -> None:
    # The server's data folder has no answer key and no resumes, yet every tool call worked.
    data_dir = runtime.settings.data_dir
    assert sorted(p.name for p in data_dir.iterdir()) == ["jobs"]
    _, blob = await every_tool_output(client, resumes)
    labels = [a for a in ARCHETYPES if a != "filler"]
    for word in (*labels, "archetype", "answer_key", "shortlist_size"):
        assert word not in blob, word


async def test_candidate_scoped_tools_never_cross_candidates(client: Client, resumes) -> None:
    ids = [r.candidate_id for r in resumes]
    for own, other in zip(ids[:15], ids[15:30], strict=True):
        other_chunks = await call(client, "get_candidate_chunks", candidate_id=other)
        bait = other_chunks[1]

        chunks = await call(client, "get_candidate_chunks", candidate_id=own)
        assert chunks and {c.candidate_id for c in chunks} == {own}
        # Even when the query is another candidate's own text.
        found = await call(client, "search_candidate", candidate_id=own, query=bait.text, k=50)
        assert found and {c.candidate_id for c in found} == {own}

        quote = bait.text.splitlines()[0]
        cross = await call(
            client, "verify_quote", candidate_id=own, chunk_id=bait.chunk_id, quote=quote
        )
        assert (cross.ok, cross.reason) == (False, "wrong_candidate")
        same = await call(
            client, "verify_quote", candidate_id=other, chunk_id=bait.chunk_id, quote=quote
        )
        assert (same.ok, same.reason) == (True, "ok")


async def test_every_answer_key_proof_passes_verify_quote(client: Client, answer_key) -> None:
    checked = 0
    for cand in answer_key.candidates:
        if not any(r.proof for r in cand.requirements):
            continue
        chunks = await call(
            client, "get_candidate_chunks", candidate_id=cand.candidate_id, sections=["experience"]
        )
        for req in cand.requirements:
            if not req.proof:
                continue
            [holder] = [c for c in chunks if req.proof in c.text]
            args = dict(candidate_id=cand.candidate_id, chunk_id=holder.chunk_id)
            assert (await call(client, "verify_quote", quote=req.proof, **args)).ok
            reworded = await call(client, "verify_quote", quote=req.proof.upper(), **args)
            assert reworded.reason == "not_verbatim"
            checked += 1
    assert checked > 100


async def test_verify_quote_reasons(client: Client, resumes) -> None:
    cid = resumes[0].candidate_id
    chunk = (await call(client, "get_candidate_chunks", candidate_id=cid))[1]
    line = chunk.text.splitlines()[1]

    async def reason(**overrides) -> str:
        args = {"candidate_id": cid, "chunk_id": chunk.chunk_id, "quote": line, **overrides}
        return (await call(client, "verify_quote", **args)).reason

    assert await reason() == "ok"
    assert await reason(quote="  " + line.replace(" ", " \n ")) == "ok"
    assert await reason(quote=line + " and more") == "not_verbatim"
    assert await reason(quote=" \n ") == "missing_quote"
    assert await reason(chunk_id=f"{cid}:experience:99") == "chunk_not_found"
    assert await reason(chunk_id="not-a-chunk-id") == "chunk_not_found"
    assert await reason(candidate_id=resumes[1].candidate_id) == "wrong_candidate"


async def test_search_resumes_shape(client: Client) -> None:
    hits = await call(client, "search_resumes", query="A/B testing and experimentation", k=30)
    assert len(hits) == 30
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 1 for s in scores)
    for hit in hits:
        assert re.fullmatch(r"cand_\d{4}", hit.candidate_id)
        assert hit.chunk_id.startswith(hit.candidate_id + ":")
        assert 0 < len(hit.snippet) <= 200 and "\n" not in hit.snippet
    skills = await call(client, "search_resumes", query="Excel", k=10, section="skills")
    assert {h.chunk_id.split(":")[1] for h in skills} == {"skills"}
    assert len(await call(client, "search_resumes", query="Excel", k=1)) == 1


async def test_jobs_and_baselines(client: Client, resumes) -> None:
    [summary] = await call(client, "list_jobs")
    assert (summary.job_id, summary.title) == (JOB_ID, "Analytics Manager")
    job = await call(client, "get_job", job_id=JOB_ID)
    assert job.job_id == JOB_ID and "## Must have" in job.text

    for tool in ("keyword_baseline", "embedding_baseline"):
        default = await call(client, tool, job_id=JOB_ID)
        assert [r.rank for r in default] == list(range(1, 21))
        everyone = await call(client, tool, job_id=JOB_ID, top_n=1000)
        assert {r.candidate_id for r in everyone} == {r.candidate_id for r in resumes}
        assert [r.candidate_id for r in everyone[:20]] == [r.candidate_id for r in default]


def hit(candidate: str, n: int, score: float) -> SearchHit:
    return SearchHit(
        candidate_id=candidate, chunk_id=f"{candidate}:experience:{n}", score=score, snippet="x"
    )


async def test_pool_stats(client: Client) -> None:
    hits = [
        hit("c1", 0, 0.95),
        hit("c1", 1, 0.10),
        hit("c2", 0, 0.31),
        hit("c3", 0, 0.29),
        hit("c4", 0, 1.0),
    ]
    stats = await call(client, "get_pool_stats", hits=[h.model_dump() for h in hits])
    assert (stats.n_hits, stats.distinct_candidates) == (5, 4)
    assert (stats.score_min, stats.score_max) == (0.10, 1.0)
    # One count per candidate, by best score: c3 -> [0.2,0.3), c2 -> [0.3,0.4), c1 and c4 -> top.
    assert [b.count for b in stats.histogram] == [0, 0, 1, 1, 0, 0, 0, 0, 0, 2]
    assert stats.histogram[0].lo == 0.0 and stats.histogram[-1].hi == 1.0
    assert (stats.strong_hit_threshold, stats.candidates_with_strong_hit) == (0.3, 3)

    empty = await call(client, "get_pool_stats", hits=[])
    assert (empty.n_hits, empty.distinct_candidates, empty.score_min) == (0, 0, None)
    unset = pool_stats(hits, None)
    assert unset.strong_hit_threshold is None and unset.candidates_with_strong_hit is None


async def test_errors_are_useful_but_do_not_leak_internals(
    client: Client, runtime: Runtime, monkeypatch
) -> None:
    with pytest.raises(ToolError, match="unknown job_id 'nope'; call list_jobs"):
        await call(client, "get_job", job_id="nope")
    for tool in ("keyword_baseline", "embedding_baseline"):
        with pytest.raises(ToolError, match="unknown job_id"):
            await call(client, tool, job_id="nope")
    for tool, extra in (("get_candidate_chunks", {}), ("search_candidate", {"query": "x"})):
        with pytest.raises(ToolError, match="unknown candidate_id 'cand_9999'"):
            await call(client, tool, candidate_id="cand_9999", **extra)
    for bad in (
        {"query": ""},
        {"query": "x", "k": 0},
        {"query": "x", "k": 101},
        {"query": "x", "section": "hobbies"},
    ):
        with pytest.raises(ToolError, match="validation error"):
            await call(client, "search_resumes", **bad)

    def explode(*args, **kwargs):
        raise RuntimeError("secret detail C:/private/path")

    monkeypatch.setattr(runtime.store, "search", explode)
    with pytest.raises(ToolError) as exc:
        await call(client, "search_resumes", query="x")
    assert "secret" not in str(exc.value) and "private" not in str(exc.value)


async def test_store_problems_are_explained(indexed, jobs_only_dir: Path, tmp_path: Path) -> None:
    cases = [
        (dict(chroma_dir=tmp_path / "empty"), "scripts/ingest.py"),
        (dict(chroma_dir=indexed[0], embedding_provider=""), "EMBEDDING_PROVIDER is not set"),
        (
            dict(
                chroma_dir=indexed[0],
                embedding_provider="openai",
                embedding_model="some-model",
                embedding_api_key="k",
            ),
            "built with embedder 'hash:bow-512' but 'openai:some-model' is configured",
        ),
    ]
    for overrides, message in cases:
        runtime = Runtime(make_settings(data_dir=jobs_only_dir, **overrides))
        assert not runtime.store_ready()
        async with Client(create_mcp(runtime)) as client:
            with pytest.raises(ToolError, match=message):
                await call(client, "search_resumes", query="x")
            # Tools that need no vector store still work.
            assert len(await call(client, "list_jobs")) == 1


async def test_response_filter_catches_what_ingest_missed(
    jobs_only_dir: Path, tmp_path: Path
) -> None:
    # Index text that never went through redaction, as if ingest had a bug.
    leaky = "Analyst, Acme (2020-01 to Present)\n- Mail me at leak@example.org or 312-555-0199"
    ResumeStore(tmp_path, HashEmbedder()).rebuild(
        [
            Chunk(
                chunk_id="cand_0001:experience:0",
                candidate_id="cand_0001",
                section="experience",
                text=leaky,
            )
        ],
        {"cand_0001": [JOB_ID]},
    )
    runtime = Runtime(make_settings(data_dir=jobs_only_dir, chroma_dir=tmp_path))
    async with Client(create_mcp(runtime)) as client:
        results = [
            await client.call_tool("get_candidate_chunks", {"candidate_id": "cand_0001"}),
            await client.call_tool(
                "search_candidate", {"candidate_id": "cand_0001", "query": "mail"}
            ),
            await client.call_tool("search_resumes", {"query": "mail"}),
        ]
    for result in results:
        blob = json.dumps(result.structured_content) + "".join(b.text for b in result.content)
        assert "leak@example.org" not in blob and "0199" not in blob
        assert blob.count(REDACTED) >= 4
        json.loads(result.content[0].text)
    assert results[0].data[0].text.endswith(f"- Mail me at {REDACTED} or {REDACTED}")


def test_scrub_result_drops_pii_fields_and_leaves_clean_results_untouched() -> None:
    payload = {
        "result": [{"candidate_id": "cand_1", "name": "Jordan Avery", "text": "see https://x.io/a"}]
    }
    dirty = ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload["result"]))],
        structured_content=payload,
    )
    clean, findings = scrub_result(dirty)
    expected = [{"candidate_id": "cand_1", "text": f"see {REDACTED}"}]
    assert clean.structured_content == {"result": expected}
    assert json.loads(clean.content[0].text) == expected
    assert len(findings) == 4 and all("Avery" not in f for f in findings)

    fine = ToolResult(
        content=[TextContent(type="text", text='{"ok":true,"reason":"ok"}')],
        structured_content={"ok": True, "reason": "ok"},
    )
    same, none = scrub_result(fine)
    assert same is fine and none == []
