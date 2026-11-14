# Ingest and MCP server

How to build the vector store from the generated resumes and run the MCP server that agents use
to read them. Covers [SPEC §5 and §6.2–6.3](./specs/SPEC.md#5-mcp-server-fastmcp).

> Synthetic data for a class project. Not for real hiring decisions.

## Contents

- [Quick start](#quick-start)
- [Embedding models](#embedding-models)
- [Ingest](#ingest)
- [Running the server](#running-the-server)
- [Tools, resources, and prompts](#tools-resources-and-prompts)
- [Using the tools from agent code](#using-the-tools-from-agent-code)
- [Guarantees and their limits](#guarantees-and-their-limits)
- [Troubleshooting](#troubleshooting)
- [Code map and tests](#code-map-and-tests)

## Quick start

No API key needed. Run from the repository root:

```powershell
uv sync
uv run python scripts/generate_data.py --offline   # data/generated/
uv run python scripts/ingest.py --offline          # chroma/
$env:EMBEDDING_PROVIDER = "hash"
uv run python -m uvicorn ats.main:app              # http://localhost:8000
```

Then:

- `GET http://localhost:8000/api/health` returns `{"status": "ok", "vector_store_loaded": true}`.
- The MCP endpoint is `http://localhost:8000/mcp/`. Point MCP Inspector or any MCP client at it.

`data/generated/`, `chroma/`, and `runs/` are all gitignored.

## Embedding models

Ingest and the server must use the **same** embedder: queries have to be turned into vectors the
same way the index was. The index records which embedder built it, and the server refuses to
search an index built with a different one.

| `EMBEDDING_PROVIDER` | What it is | Use it for |
|---|---|---|
| `openai` | Any OpenAI-compatible `/embeddings` endpoint. Needs `EMBEDDING_MODEL` and `EMBEDDING_API_KEY`; `EMBEDDING_BASE_URL` changes the host. | Real runs and every reported result |
| `hash` | Built in, offline, deterministic. Matches on shared (stemmed) words only. | Tests, CI, and wiring things up locally |

`scripts/ingest.py --offline` is a shortcut for the `hash` embedder.

**Do not report results from the `hash` embedder.** It has no notion of meaning, so "led a
5-person group" does not match "people management". On the offline dataset its embedding
baseline picks the same kinds of candidates as the keyword baseline (every keyword stuffer, no
hidden gem). The project's main comparison needs a real embedding model.

The `openai` embedder is tested against a mock server only. It has not yet been run against a
live provider.

## Ingest

```powershell
uv run python scripts/ingest.py [--offline] [--data-dir DIR] [--chroma-dir DIR]
```

For each resume in `data/generated/resumes/`:

1. **Chunk.** One chunk for the summary, each experience entry, education, the skills list, and
   each project. Bullets are never split. A block over ~120 tokens is split between bullets and
   its header line is repeated. Chunk ids are `{candidate_id}:{section}:{n}`.
2. **Redact.** The `pii` field is never read into chunk text. As a second check, the candidate's
   own name, email, phone, address, and date of birth are replaced with `[REDACTED]` if they
   appear, along with anything shaped like an email, phone number, or URL.
3. **Embed and index** into Chroma: collection `resumes` (one row per chunk) and `resume_docs`
   (one row per candidate, for the baselines).

An experience chunk looks like this:

```
Data Analyst, Tidewater Logistics Tech (2023-04 to Present)
- Reviewed pull requests for the shared metrics repository
- Built cohort retention curves that changed how the product group ran onboarding
- Provided people management for 8 direct reports across the analytics and data engineering functions
```

Ingest prints a summary and exits non-zero if something is wrong:

```
Indexed 50 resumes (339 chunks, largest ~116 tokens) for 1 job(s) into chroma (embedder: hash:bow-512)
Every answer-key proof line is in the index word for word.
```

- **Proof-line check.** If `answer_key.json` is present, every proof line must still be a whole,
  unchanged bullet in the index. If one is not, quote checks and the eval would be wrong, so
  ingest fails.
- **Redaction warnings.** `WARNING: redacted name from cand_0007` means PII was found in resume
  text. Generator output should never trigger this.
- **Re-running** replaces the whole index. Everything is embedded first, so if the embedding
  provider fails midway, the previous index is left as it was.

## Running the server

```powershell
uv run python -m uvicorn ats.main:app
```

Settings come from the environment or a `.env` file (see `.env.example`). The ones that matter here:

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY` | none | Must match what ingest used |
| `DATA_DIR` | `data/generated` | Where `jobs/*.json` is read from |
| `CHROMA_DIR` | `chroma` | The vector store |
| `RUNS_DIR` | `runs` | Where run traces are written |
| `MCP_API_KEY` | unset | When set, `/mcp` requires `Authorization: Bearer <key>` |
| `STRONG_HIT` | unset | Score threshold that `get_pool_stats` counts as a strong hit |

Relative paths are resolved from the folder you start the server in.

**Always set `MCP_API_KEY` on a public deployment.** Without it, anyone who can reach the URL can
read every resume chunk. `/api/health` is always open.

## Tools, resources, and prompts

| Tool | Arguments | Returns |
|---|---|---|
| `list_jobs` | none | `list[JobSummary]` |
| `get_job` | `job_id` | `JobDescription` (with the full markdown) |
| `search_resumes` | `query`, `k=20` (1–100), `section=None` | `list[SearchHit]`: candidate id, chunk id, score, and a snippet of at most 200 characters |
| `get_pool_stats` | `hits: list[SearchHit]` | `PoolStats`: distinct candidates, score range, a 10-bucket histogram of each candidate's best score, and how many candidates have a strong hit |
| `get_candidate_chunks` | `candidate_id`, `sections=None` | `list[Chunk]` in resume order |
| `search_candidate` | `candidate_id`, `query`, `k=5` (1–50) | `list[Chunk]`, best match first |
| `verify_quote` | `candidate_id`, `chunk_id`, `quote` | `QuoteCheck`: `ok` plus a reason |
| `keyword_baseline` | `job_id`, `top_n=20` | `list[RankedCandidate]` by BM25 |
| `embedding_baseline` | `job_id`, `top_n=20` | `list[RankedCandidate]` by whole-resume similarity |

- **Scores** from the search tools and `embedding_baseline` are cosine similarity from 0 to 1.
  `keyword_baseline` scores are raw BM25 and are not on that scale.
- **`get_pool_stats` takes the hits themselves**, not a list of ids as the SPEC table first said.
  It needs the scores to build a histogram. Pass what `search_resumes` returned.
- **`verify_quote` reasons:** `ok`, `chunk_not_found`, `wrong_candidate`, `missing_quote`,
  `not_verbatim`. Only whitespace and curly/straight quote marks are normalized. Matching is
  case-sensitive, and a quote may not join text from two chunks.
- **Resources:** `ats://jobs/{job_id}` (JD markdown) and `ats://runs/{run_id}/trace` (the run's
  trace as JSON Lines).
- **Prompts:** `requirements_system`, `evaluator_system`, `verifier_system`, read from
  `src/ats/agents/prompts/*.md`. The current files are first drafts written from SPEC §4; the
  agent owners should replace them.

## Using the tools from agent code

In tests, pass the server object straight to the client. Nothing has to be running:

```python
from fastmcp import Client
from ats.config import Settings
from ats.mcp_server import Runtime, create_mcp

runtime = Runtime(Settings(embedding_provider="hash", chroma_dir=..., data_dir=...))
async with Client(create_mcp(runtime)) as client:
    found = await client.call_tool("search_resumes", {"query": "people management", "k": 20})
    hits = found.data  # typed objects: hit.candidate_id, hit.score
    stats = await client.call_tool("get_pool_stats", {"hits": found.structured_content["result"]})
```

The `tests/conftest.py` fixtures `dataset` and `indexed` give you a generated 50-resume pool and
a built index to point `data_dir` and `chroma_dir` at.

Against a running server, send the bearer token and, to get tool calls into a run's trace, the
run id:

```python
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    settings.mcp_url, headers={"X-Run-Id": run_id}, auth=settings.mcp_api_key.get_secret_value()
)
async with Client(transport) as client:
    ...
```

Things to know when calling:

- **Create `runs/{run_id}/` before the first tool call.** Tool calls are only traced for runs
  whose folder exists. Run ids may use letters, digits, `_` and `-` (64 characters at most).
- **Errors** arrive as `fastmcp.exceptions.ToolError`. Unknown ids, bad arguments, a missing index,
  and an embedder mismatch have readable messages. Any other failure is reported as a generic
  error, with details only in the server log.
- Each traced call is one line in `runs/{run_id}/trace.jsonl` with `stage: "mcp"`,
  `event: "tool_call"`, the tool name, its arguments (long values shortened), whether it
  succeeded, the size of the result, and the latency.

## Guarantees and their limits

| Guarantee (SPEC §5.3) | How it is enforced | What it does not cover |
|---|---|---|
| No tool returns `name`, `email`, `phone`, `address`, `photo_url`, or `dob` | Redaction at ingest, then a filter on every tool result that removes fields with those names and redacts email-, phone-, and URL-shaped text | The name of a *different* person written inside a bullet. Only a candidate's own recorded name can be recognised. |
| Candidate-scoped tools never return another candidate's chunks | The store query filters by candidate, and the tool drops anything else. `verify_quote` answers `wrong_candidate` before looking at the text. | |
| Every tool call is logged to the run's trace | A server-wide hook, so new tools are covered automatically | Calls with no `X-Run-Id`, an invalid one, or one whose run folder does not exist are not logged |
| The answer key is never reachable through MCP | The server only reads `jobs/*.json` and the index. The contract tests run it with no answer key on disk. | |

## Troubleshooting

| Message | What to do |
|---|---|
| `EMBEDDING_PROVIDER is not set` | Set it to `openai` or `hash` for the server, or pass `--offline` to ingest |
| `no vector store at ...` / `has no data` | Run `scripts/ingest.py` |
| `the index was built with embedder 'X' but 'Y' is configured` | Re-run ingest, or change `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` to match the index |
| `no jobs or resumes under ...` | Run `scripts/generate_data.py` first |
| `... proof line not found verbatim` | The dataset and index disagree. Regenerate the data and re-ingest. |
| `embedding request rejected (HTTP 401)` | Check `EMBEDDING_API_KEY` |
| `401 {"error": "unauthorized"}` from `/mcp` | Send `Authorization: Bearer <MCP_API_KEY>` |
| Tool calls missing from a trace | Create `runs/{run_id}/` first and send `X-Run-Id` |

## Code map and tests

| File | Role |
|---|---|
| [scripts/ingest.py](../scripts/ingest.py) | Ingest CLI |
| [src/ats/rag/chunking.py](../src/ats/rag/chunking.py) | Section-aware chunks |
| [src/ats/rag/redaction.py](../src/ats/rag/redaction.py) | Redaction at ingest and the tool response filter |
| [src/ats/rag/embeddings.py](../src/ats/rag/embeddings.py) | `Embedder` interface, `openai` and `hash` embedders |
| [src/ats/rag/store.py](../src/ats/rag/store.py) | Chroma collections, search, embedder check |
| [src/ats/rag/ingest.py](../src/ats/rag/ingest.py) | Chunk → redact → index, and the proof-line check |
| [src/ats/rag/quotes.py](../src/ats/rag/quotes.py) | The rules behind `verify_quote` |
| [src/ats/rag/baselines.py](../src/ats/rag/baselines.py) | BM25 and embedding ranking |
| [src/ats/mcp_server.py](../src/ats/mcp_server.py) | Tools, resources, prompts, response filter, trace hook |
| [src/ats/main.py](../src/ats/main.py) | FastAPI app, `/mcp` mount, bearer check |
| [src/ats/tracing.py](../src/ats/tracing.py) | Trace file writer |
| [src/ats/config.py](../src/ats/config.py) | Settings |

```powershell
uv run python -m pytest -q
uv run ruff check .
uv run ruff format --check .
```

The tests need no network and no keys. They use the offline generator and the `hash` embedder.
