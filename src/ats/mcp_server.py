"""FastMCP server ``agentic-ats``: the only data access path for agents (SPEC §5).

Guarantees enforced here rather than left to each tool (SPEC §5.3):
1. Tool output passes a response filter that drops PII-named fields and redacts PII-shaped text.
2. Candidate-scoped tools only ever return that candidate's chunks.
3. Every tool call is appended to the run's trace when the request carries ``X-Run-Id``.

Nothing in this module reads the answer key.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware
from fastmcp.tools.tool import ToolResult
from pydantic import Field

from ats.config import Settings, get_settings
from ats.data import load_jobs
from ats.models import (
    Chunk,
    JobDescription,
    JobSummary,
    PoolStats,
    QuoteCheck,
    RankedCandidate,
    ScoreBucket,
    SearchHit,
    Section,
    TraceEvent,
)
from ats.rag import baselines
from ats.rag.embeddings import EmbeddingConfigError, EmbeddingError, embedder_from_settings
from ats.rag.quotes import check_quote
from ats.rag.redaction import redact_text, scrub
from ats.rag.store import ResumeStore, StoreError
from ats.run_store import FileRunStore
from ats.tracing import append_event, read_trace, valid_run_id

log = logging.getLogger(__name__)

SERVER_NAME = "agentic-ats"
SNIPPET_CHARS = 200
PROMPTS_DIR = Path(__file__).parent / "agents" / "prompts"
RUN_ID_HEADER = "x-run-id"

Query = Annotated[str, Field(min_length=1, max_length=2000)]
READ_ONLY = {"readOnlyHint": True}


class Runtime:
    """Settings plus services that are built on first use."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._store: ResumeStore | None = None
        self._jobs: dict[str, JobDescription] = {}

        # File-based persistence for screening run state.
        self.run_store = FileRunStore(settings.runs_dir)

    @property
    def store(self) -> ResumeStore:
        with self._lock:
            if self._store is None:
                self._store = ResumeStore(
                    self.settings.chroma_dir, embedder_from_settings(self.settings)
                )
            return self._store

    @property
    def jobs(self) -> dict[str, JobDescription]:
        with self._lock:
            if not self._jobs:
                self._jobs = load_jobs(self.settings.data_dir)
            return self._jobs

    def store_ready(self) -> bool:
        try:
            return self.store.is_ready()
        except (EmbeddingError, StoreError):
            return False


def pool_stats(hits: list[SearchHit], strong_hit: float | None) -> PoolStats:
    best: dict[str, float] = {}
    for hit in hits:
        best[hit.candidate_id] = max(best.get(hit.candidate_id, 0.0), hit.score)
    counts = [0] * 10
    for score in best.values():
        counts[min(9, max(0, int(score * 10)))] += 1
    scores = [hit.score for hit in hits]
    return PoolStats(
        n_hits=len(hits),
        distinct_candidates=len(best),
        score_min=min(scores) if scores else None,
        score_max=max(scores) if scores else None,
        histogram=[ScoreBucket(lo=i / 10, hi=(i + 1) / 10, count=n) for i, n in enumerate(counts)],
        strong_hit_threshold=strong_hit,
        candidates_with_strong_hit=(
            None if strong_hit is None else sum(s >= strong_hit for s in best.values())
        ),
    )


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


@contextmanager
def _tool_errors():
    """Turn expected failures into messages the caller can act on."""
    try:
        yield
    except (StoreError, EmbeddingConfigError) as exc:
        raise ToolError(str(exc)) from exc
    except EmbeddingError as exc:
        log.error("embedding request failed: %s", exc)
        raise ToolError("the embedding provider request failed; see the server log") from exc


def _compact(value: Any) -> Any:
    """Keep traced arguments small."""
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + f"... ({len(value)} chars)"
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, list):
        head = [_compact(v) for v in value[:20]]
        return head if len(value) <= 20 else [*head, f"... ({len(value)} items)"]
    return value


def scrub_result(result: ToolResult) -> tuple[ToolResult, list[str]]:
    """Apply the response filter to a tool result. Returns the result unchanged when clean."""
    findings: list[str] = []
    structured = result.structured_content
    if structured is not None:
        structured, found = scrub(structured)
        findings += found
    blocks = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            try:
                clean, found = scrub(json.loads(text))
                new_text = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
            except json.JSONDecodeError:
                new_text, found = redact_text(text)
            if found:
                findings += found
                block = block.model_copy(update={"text": new_text})
        blocks.append(block)
    if not findings:
        return result, []
    return ToolResult(content=blocks, structured_content=structured, meta=result.meta), findings


class GuardMiddleware(Middleware):
    """Runs around every tool call: response filter, then trace logging."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    async def on_call_tool(self, context, call_next):
        name = context.message.name
        args = context.message.arguments or {}
        started = time.perf_counter()
        result: ToolResult | None = None
        redactions = 0
        error: str | None = None
        try:
            result = await call_next(context)
            result, findings = scrub_result(result)
            if findings:
                redactions = len(findings)
                log.warning("response filter changed the output of %s: %s", name, findings)
            return result
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self._trace(name, args, result, error, redactions, started)

    def _trace(self, name, args, result, error, redactions, started) -> None:
        run_id = get_http_headers().get(RUN_ID_HEADER)
        if not run_id:
            return
        if not valid_run_id(run_id):
            log.warning("ignored a malformed %s header", RUN_ID_HEADER)
            return
        try:
            chars = (
                sum(len(getattr(b, "text", "") or "") for b in (result.content or []))
                if result
                else 0
            )
            written = append_event(
                self._runtime.settings.runs_dir,
                TraceEvent(
                    ts=datetime.now(UTC),
                    run_id=run_id,
                    stage="mcp",
                    agent=None,
                    event="tool_call",
                    round=None,
                    tool=name,
                    input=_compact(args),
                    output={
                        "ok": error is None,
                        "error": error,
                        "result_chars": chars,
                        "redactions": redactions,
                    },
                    latency_ms=int((time.perf_counter() - started) * 1000),
                ),
            )
            if not written:
                log.info("no run folder for %s; tool call not traced", run_id)
        except Exception:
            log.exception("could not write the trace event for %s", name)


def create_mcp(runtime: Runtime) -> FastMCP:
    mcp = FastMCP(
        SERVER_NAME,
        instructions=(
            "Tools for screening synthetic resumes against a job description. Candidates are "
            "identified only by anonymized ids; no tool returns personal details."
        ),
        middleware=[GuardMiddleware(runtime)],
        mask_error_details=True,
    )

    async def with_store(fn):
        """Run ``fn(store)`` off the event loop."""
        with _tool_errors():
            return await anyio.to_thread.run_sync(partial(fn, runtime.store))

    def job_or_error(job_id: str) -> JobDescription:
        job = runtime.jobs.get(job_id)
        if job is None:
            raise ToolError(f"unknown job_id {job_id!r}; call list_jobs to see valid ids")
        return job

    async def require_candidate(candidate_id: str) -> None:
        if not await with_store(lambda s: s.has_candidate(candidate_id)):
            raise ToolError(f"unknown candidate_id {candidate_id!r}")

    def own_chunks(candidate_id: str, chunks: list[Chunk]) -> list[Chunk]:
        """Second check on candidate scoping; the store query already filters."""
        own = [c for c in chunks if c.candidate_id == candidate_id]
        if len(own) != len(chunks):
            log.error("dropped %d chunk(s) of another candidate", len(chunks) - len(own))
        return own

    # --- jobs ---------------------------------------------------------------------------------

    @mcp.tool(annotations=READ_ONLY)
    async def list_jobs() -> list[JobSummary]:
        """List the available job descriptions."""
        return [JobSummary(job_id=j.job_id, title=j.title) for j in runtime.jobs.values()]

    @mcp.tool(annotations=READ_ONLY)
    async def get_job(job_id: str) -> JobDescription:
        """Get one job description, including its full markdown text."""
        return job_or_error(job_id)

    # --- search across the pool ---------------------------------------------------------------

    @mcp.tool(annotations=READ_ONLY)
    async def search_resumes(
        query: Query,
        k: Annotated[int, Field(ge=1, le=100)] = 20,
        section: Section | None = None,
    ) -> list[SearchHit]:
        """Vector search over all resume chunks. Scores are cosine similarity from 0 to 1."""
        hits = await with_store(lambda s: s.search(query, k, section=section))
        return [
            SearchHit(
                candidate_id=chunk.candidate_id,
                chunk_id=chunk.chunk_id,
                score=round(score, 6),
                snippet=" ".join(chunk.text.split())[:SNIPPET_CHARS],
            )
            for chunk, score in hits
        ]

    @mcp.tool(annotations=READ_ONLY)
    async def get_pool_stats(hits: list[SearchHit]) -> PoolStats:
        """Summarize search hits and strong-hit counts."""
        return pool_stats(hits, runtime.settings.strong_hit)

    # --- one candidate ------------------------------------------------------------------------

    @mcp.tool(annotations=READ_ONLY)
    async def get_candidate_chunks(
        candidate_id: str, sections: list[Section] | None = None
    ) -> list[Chunk]:
        """All chunks of one candidate's resume, in resume order, with personal details removed."""
        await require_candidate(candidate_id)
        chunks = await with_store(lambda s: s.candidate_chunks(candidate_id, sections))
        return own_chunks(candidate_id, chunks)

    @mcp.tool(annotations=READ_ONLY)
    async def search_candidate(
        candidate_id: str,
        query: Query,
        k: Annotated[int, Field(ge=1, le=50)] = 5,
    ) -> list[Chunk]:
        """Vector search restricted to one candidate's resume, best match first."""
        await require_candidate(candidate_id)
        hits = await with_store(lambda s: s.search(query, k, candidate_id=candidate_id))
        return own_chunks(candidate_id, [chunk for chunk, _ in hits])

    @mcp.tool(annotations=READ_ONLY)
    async def verify_quote(candidate_id: str, chunk_id: str, quote: str) -> QuoteCheck:
        """Check that a quote appears word for word in the given candidate's chunk."""
        chunk = await with_store(lambda s: s.get_chunk(chunk_id))
        return check_quote(chunk, candidate_id, quote)

    # --- baselines ----------------------------------------------------------------------------

    @mcp.tool(annotations=READ_ONLY)
    async def keyword_baseline(
        job_id: str, top_n: Annotated[int, Field(ge=1, le=1000)] = 20
    ) -> list[RankedCandidate]:
        """Rank a job's candidate pool by BM25 keyword score of the whole resume."""
        job = job_or_error(job_id)
        return await with_store(lambda s: baselines.keyword_baseline(s, job_id, job.text, top_n))

    @mcp.tool(annotations=READ_ONLY)
    async def embedding_baseline(
        job_id: str, top_n: Annotated[int, Field(ge=1, le=1000)] = 20
    ) -> list[RankedCandidate]:
        """Rank a job's candidate pool by cosine similarity of the whole resume to the job."""
        job = job_or_error(job_id)
        return await with_store(lambda s: baselines.embedding_baseline(s, job_id, job.text, top_n))

    # --- resources ----------------------------------------------------------------------------

    @mcp.resource("ats://jobs/{job_id}", mime_type="text/markdown")
    async def job_markdown(job_id: str) -> str:
        """The job description as markdown."""
        job = runtime.jobs.get(job_id)
        if job is None:
            raise ResourceError(f"unknown job_id {job_id!r}")
        return redact_text(job.text)[0]

    @mcp.resource("ats://runs/{run_id}/trace", mime_type="application/x-ndjson")
    async def run_trace(run_id: str) -> str:
        """A run's trace as JSON Lines, one event per line."""
        trace = read_trace(runtime.settings.runs_dir, run_id) if valid_run_id(run_id) else None
        if trace is None:
            raise ResourceError("no trace for that run id")
        return trace

    # --- prompts ------------------------------------------------------------------------------

    @mcp.prompt(name="requirements_system", description="System prompt for the Requirements agent")
    def requirements_system() -> str:
        return load_prompt("requirements_system")

    @mcp.prompt(name="evaluator_system", description="System prompt for the Evaluator agent")
    def evaluator_system() -> str:
        return load_prompt("evaluator_system")

    @mcp.prompt(name="verifier_system", description="System prompt for the Verifier agent")
    def verifier_system() -> str:
        return load_prompt("verifier_system")

    return mcp


runtime = Runtime(get_settings())
mcp = create_mcp(runtime)
