"""Tests for the adaptive MCP Search Agent."""

from types import SimpleNamespace

import pytest

from ats.agents.search import (
    decide_search_strategy,
    generate_search_queries,
    run_search_agent,
)
from ats.config import Settings
from ats.models import Requirement, RequirementSet


class FakeMCPClient:
    """Mock MCP client that always returns two strong-hit candidates."""

    def __init__(self):
        self.search_calls = []
        self.stats_calls = []

    async def call_tool(self, tool_name, arguments):
        if tool_name == "search_resumes":
            self.search_calls.append(arguments["query"])

            return SimpleNamespace(
                is_error=False,
                data=[
                    {
                        "candidate_id": "C001",
                        "chunk_id": "CH001",
                        "score": 0.90,
                        "snippet": "SQL experience",
                    },
                    {
                        "candidate_id": "C002",
                        "chunk_id": "CH002",
                        "score": 0.85,
                        "snippet": "Python experience",
                    },
                ],
                structured_content=None,
            )

        if tool_name == "get_pool_stats":
            self.stats_calls.append(arguments)

            return SimpleNamespace(
                is_error=False,
                data={
                    "n_hits": len(arguments["hits"]),
                    "distinct_candidates": 2,
                    "score_min": 0.85,
                    "score_max": 0.90,
                    "histogram": [],
                    "strong_hit_threshold": 0.80,
                    "candidates_with_strong_hit": 2,
                },
                structured_content=None,
            )

        raise ValueError(f"Unexpected MCP tool: {tool_name}")


@pytest.fixture
def requirements():
    return RequirementSet(
        job_id="JOB-001",
        requirements=[
            Requirement(
                id="R1",
                text="SQL",
                kind="must",
                weight=3,
                equivalents=["PostgreSQL"],
            ),
            Requirement(
                id="R2",
                text="Python",
                kind="must",
                weight=3,
                equivalents=["Python programming"],
            ),
        ],
    )


@pytest.fixture
def settings():
    return Settings(
        max_search_rounds=3,
        max_pool_size=20,
        strong_hit=0.80,
    )


def test_initial_search_query(requirements):
    queries = generate_search_queries(requirements)

    assert queries == ["SQL Python"]


def test_broaden_when_too_few_candidates():
    assert decide_search_strategy(2) == "broaden"


def test_stop_when_candidate_count_is_in_target_range():
    assert decide_search_strategy(15) == "stop"


def test_narrow_when_too_many_candidates():
    assert decide_search_strategy(25) == "narrow"


@pytest.mark.asyncio
async def test_search_stops_after_maximum_three_rounds(
    requirements,
    settings,
):
    mcp_client = FakeMCPClient()

    result = await run_search_agent(
        requirements,
        mcp_client,
        settings=settings,
    )

    assert len(result.queries) == 3
    assert len(result.queries) <= settings.max_search_rounds

    print("Rounds completed:", len(result.queries))
    print("Search calls:", mcp_client.search_calls)
    print("Candidate IDs:", result.candidate_ids)

    print("PASS: Search stopped after the maximum 3 rounds.")


@pytest.mark.asyncio
async def test_search_returns_two_distinct_candidates(
    requirements,
    settings,
):
    mcp_client = FakeMCPClient()

    result = await run_search_agent(
        requirements,
        mcp_client,
        settings=settings,
    )

    assert result.candidate_ids == ["C001", "C002"]


@pytest.mark.asyncio
async def test_search_broadens_after_finding_too_few_candidates(
    requirements,
    settings,
):
    mcp_client = FakeMCPClient()

    result = await run_search_agent(
        requirements,
        mcp_client,
        settings=settings,
    )

    assert len(result.queries) == 3

    # First round uses the combined initial query.
    assert result.queries[0].query == "SQL Python"

    # Later rounds broaden to individual requirements and equivalents.
    assert "SQL" in mcp_client.search_calls
    assert "PostgreSQL" in mcp_client.search_calls
    assert "Python programming" in mcp_client.search_calls


@pytest.mark.asyncio
async def test_search_uses_mcp_tools(
    requirements,
    settings,
):
    mcp_client = FakeMCPClient()

    await run_search_agent(
        requirements,
        mcp_client,
        settings=settings,
    )

    assert len(mcp_client.search_calls) > 0
    assert len(mcp_client.stats_calls) == 3


@pytest.mark.asyncio
async def test_search_rejects_empty_requirements(settings):
    mcp_client = FakeMCPClient()

    empty_requirements = RequirementSet(
        job_id="JOB-001",
        requirements=[],
    )

    with pytest.raises(ValueError, match="No requirements"):
        await run_search_agent(
            empty_requirements,
            mcp_client,
            settings=settings,
        )


async def test_search_accepts_fastmcp_dataclass_results(requirements, settings):
    """FastMCP returns hits and pool stats as generated dataclass objects."""
    from dataclasses import make_dataclass

    Hit = make_dataclass(
        "Root",
        [("candidate_id", str), ("chunk_id", str), ("score", float), ("snippet", str)],
    )
    inner = FakeMCPClient()

    class DataclassMCPClient:
        async def call_tool(self, tool_name, arguments):
            result = await inner.call_tool(tool_name, arguments)
            if tool_name == "search_resumes":
                result.data = [Hit(**hit) for hit in result.data]
            if tool_name == "get_pool_stats":
                Stats = make_dataclass("Root", list(result.data))
                result.data = Stats(**result.data)
            return result

    pool = await run_search_agent(requirements, DataclassMCPClient(), settings=settings)

    assert pool.candidate_ids == ["C001", "C002"]
