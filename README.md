# Agentic ATS

> **Synthetic data. Class project. Not for real hiring decisions.**

Given one job description and a pool of resumes, Agentic ATS produces a ranked shortlist in which
every "meets requirement" claim cites a line copied word for word from the resume. A human
recruiter makes the final call, and nothing is ever sent to a candidate.

The project asks whether a small multi-agent system beats keyword and embedding baselines at
finding qualified candidates who use different wording ("hidden gems") and at rejecting resumes
that only list keywords. The design is in [docs/specs/SPEC.md](docs/specs/SPEC.md).

## Status

| Part | State |
|---|---|
| Synthetic data generator | Built. [Guide](docs/data-generator.md) |
| Ingest, vector store, baselines | Built. [Guide](docs/mcp-server.md) |
| MCP server (tools, resources, prompts) | Built. [Guide](docs/mcp-server.md) |
| Agents and orchestrator | Not started |
| REST API | `/api/health` only |
| Streamlit UI | Not started |
| Evaluation metrics and report | Not started |
| Deployment | Not deployed; no live URLs yet |

## Run it locally

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12. No API key is needed for these steps.

```powershell
uv sync
uv run python scripts/generate_data.py --offline   # 1 job, 50 resumes, answer key -> data/generated/
uv run python scripts/ingest.py --offline          # vector store -> chroma/
$env:EMBEDDING_PROVIDER = "hash"                   # bash: export EMBEDDING_PROVIDER=hash
uv run python -m uvicorn ats.main:app              # http://localhost:8000
```

- Health check: `http://localhost:8000/api/health`
- MCP endpoint: `http://localhost:8000/mcp/` (connect MCP Inspector or any MCP client)

`--offline` uses template resumes and a word-matching embedder so everything runs without keys.
That is enough to develop against, but not to draw conclusions from: real runs need an LLM for the
generator and a real embedding model. See the two guides for how to configure them.

## Configuration and secrets

Copy `.env.example` to `.env` and fill in what you need. `.env` is gitignored; never commit keys.
Set `MCP_API_KEY` on any public deployment, or the resume data is readable by anyone with the URL.

## Tests

```powershell
uv run python -m pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests need no network and no keys. CI runs the same three commands on every pull request.

## Team

Abdullah Rafiq, Sairamya Macha, Noah Velez. Ownership and milestones are in
[SPEC §16](docs/specs/SPEC.md#16-ownership-and-milestones).
