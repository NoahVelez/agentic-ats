from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from ats.config import Settings
from ats.models import TraceEvent
from ats.tracing import append_event, read_trace, run_dir, valid_run_id

ENV_VARS = (
    "DATA_DIR",
    "CHROMA_DIR",
    "RUNS_DIR",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_API_KEY",
    "MCP_API_KEY",
    "MAX_SEARCH_ROUNDS",
    "STRONG_HIT",
    "EVAL_CONCURRENCY",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path: Path):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def test_settings_defaults_match_the_spec(clean_env) -> None:
    s = Settings()
    assert (s.max_search_rounds, s.max_verify_rounds, s.max_clarify_questions) == (3, 2, 3)
    assert (s.max_pool_size, s.run_token_budget, s.run_cost_budget_usd) == (20, 300_000, 1.0)
    assert s.eval_concurrency == 5 and s.strong_hit is None
    assert s.data_dir == Path("data/generated") and s.chroma_dir == Path("chroma")
    assert s.runs_dir == Path("runs") and s.mcp_url == "http://localhost:8000/mcp/"
    assert s.mcp_api_key is None and s.embedding_provider == ""


def test_settings_read_env_and_dotenv_and_ignore_empty_values(clean_env, tmp_path: Path) -> None:
    # A .env copied from .env.example has empty values; those must not break number parsing.
    (tmp_path / ".env").write_text(
        "MAX_SEARCH_ROUNDS=\nSTRONG_HIT=\nEMBEDDING_PROVIDER=hash\nMCP_API_KEY=from-dotenv\n",
        encoding="utf-8",
    )
    clean_env.setenv("EVAL_CONCURRENCY", "8")
    clean_env.setenv("MCP_API_KEY", "from-env")
    s = Settings()
    assert s.max_search_rounds == 3 and s.strong_hit is None
    assert s.embedding_provider == "hash" and s.eval_concurrency == 8
    assert s.mcp_api_key.get_secret_value() == "from-env"
    assert "from-env" not in repr(s) and "from-env" not in str(s.model_dump())


def test_settings_reject_out_of_range_values(clean_env) -> None:
    clean_env.setenv("STRONG_HIT", "1.5")
    with pytest.raises(ValidationError):
        Settings()


def event(run_id: str, tool: str = "search_resumes") -> TraceEvent:
    return TraceEvent(
        ts=datetime(2026, 11, 14, tzinfo=UTC),
        run_id=run_id,
        stage="mcp",
        agent=None,
        event="tool_call",
        round=None,
        tool=tool,
        input={"query": "x"},
        output={"ok": True},
        latency_ms=3,
    )


@pytest.mark.parametrize("run_id", ["run_1", "A-b_9", "x" * 64])
def test_valid_run_ids(run_id: str) -> None:
    assert valid_run_id(run_id)


@pytest.mark.parametrize(
    "run_id", ["", "../x", "a/b", "a\\b", "..", ".hidden", "x" * 65, "a b", "a\n", "_x"]
)
def test_invalid_run_ids_are_rejected_everywhere(run_id: str, tmp_path: Path) -> None:
    assert not valid_run_id(run_id)
    with pytest.raises(ValueError):
        run_dir(tmp_path, run_id)
    with pytest.raises(ValueError):
        read_trace(tmp_path, run_id)
    with pytest.raises(ValueError):
        append_event(tmp_path, event(run_id))
    assert list(tmp_path.iterdir()) == []


def test_events_are_appended_only_for_existing_runs(tmp_path: Path) -> None:
    assert append_event(tmp_path, event("run_1")) is False
    assert list(tmp_path.iterdir()) == [] and read_trace(tmp_path, "run_1") is None

    (tmp_path / "run_1").mkdir()
    assert append_event(tmp_path, event("run_1")) is True
    assert append_event(tmp_path, event("run_1", tool="get_job")) is True
    lines = read_trace(tmp_path, "run_1").splitlines()
    assert [TraceEvent.model_validate_json(line).tool for line in lines] == [
        "search_resumes",
        "get_job",
    ]
