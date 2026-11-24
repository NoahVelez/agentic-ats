
import json
from types import SimpleNamespace

import pytest

from ats.agents.evaluator import (
    parse_model_response,
    run_evaluator_agent,
)
from ats.models import Requirement, RequirementSet


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def complete(self, system, user):
        self.calls.append({"system": system, "user": user})
        return self.response


class FakeMCPClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))

        if name == "get_candidate_chunks":
            return SimpleNamespace(
                is_error=False,
                data=self.chunks,
                structured_content=None,
            )

        if name == "search_candidate":
            return SimpleNamespace(
                is_error=False,
                data=self.chunks,
                structured_content=None,
            )

        return SimpleNamespace(
            is_error=True,
            data=None,
            structured_content=None,
        )


@pytest.fixture
def requirements():
    return RequirementSet(
        job_id="job_001",
        requirements=[
            Requirement(
                id="R1",
                text="Experience with Python",
                kind="must",
                weight=3,
                equivalents=[],
                assumption=None,
            ),
            Requirement(
                id="R2",
                text="Experience with SQL",
                kind="nice",
                weight=2,
                equivalents=[],
                assumption=None,
            ),
        ],
    )


@pytest.fixture
def chunks():
    return [
        {
            "chunk_id": "chunk_001",
            "candidate_id": "candidate_001",
            "section": "experience",
            "text": "Developed Python applications for data analysis.",
        },
        {
            "chunk_id": "chunk_002",
            "candidate_id": "candidate_001",
            "section": "projects",
            "text": "Used SQL to query and analyze business data.",
        },
    ]


def valid_evaluation():
    return {
        "candidate_id": "candidate_001",
        "claims": [
            {
                "requirement_id": "R1",
                "verdict": "met",
                "quote": "Developed Python applications for data analysis.",
                "chunk_id": "chunk_001",
                "reasoning": "The candidate used Python in data analysis.",
            },
            {
                "requirement_id": "R2",
                "verdict": "met",
                "quote": "Used SQL to query and analyze business data.",
                "chunk_id": "chunk_002",
                "reasoning": "The candidate used SQL to analyze data.",
            },
        ],
    }


def test_parse_model_response_valid_json():
    result = parse_model_response(
        '{"candidate_id": "candidate_001", "claims": []}'
    )
    assert result["candidate_id"] == "candidate_001"


def test_parse_model_response_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_model_response("not valid JSON")


@pytest.mark.asyncio
async def test_evaluator_returns_candidate_evaluation(requirements, chunks):
    llm = FakeLLMClient(json.dumps(valid_evaluation()))
    mcp = FakeMCPClient(chunks)

    result = await run_evaluator_agent(
        requirements=requirements,
        candidate_id="candidate_001",
        mcp_client=mcp,
        llm_client=llm,
    )

    assert result.candidate_id == "candidate_001"
    assert len(result.claims) == 2
    assert result.claims[0].verdict == "met"
    assert result.claims[1].verdict == "met"


@pytest.mark.asyncio
async def test_evaluator_calls_required_mcp_tools(requirements, chunks):
    llm = FakeLLMClient(json.dumps(valid_evaluation()))
    mcp = FakeMCPClient(chunks)

    await run_evaluator_agent(
        requirements=requirements,
        candidate_id="candidate_001",
        mcp_client=mcp,
        llm_client=llm,
    )

    called_tools = [call[0] for call in mcp.calls]

    assert "get_candidate_chunks" in called_tools
    assert "search_candidate" in called_tools


@pytest.mark.asyncio
async def test_evaluator_rejects_wrong_candidate_id(requirements, chunks):
    evaluation = valid_evaluation()
    evaluation["candidate_id"] = "different_candidate"

    llm = FakeLLMClient(json.dumps(evaluation))
    mcp = FakeMCPClient(chunks)

    with pytest.raises(ValueError):
        await run_evaluator_agent(
            requirements=requirements,
            candidate_id="candidate_001",
            mcp_client=mcp,
            llm_client=llm,
        )


@pytest.mark.asyncio
async def test_evaluator_rejects_missing_requirement_claim(requirements, chunks):
    evaluation = valid_evaluation()
    evaluation["claims"].pop()

    llm = FakeLLMClient(json.dumps(evaluation))
    mcp = FakeMCPClient(chunks)

    with pytest.raises(ValueError):
        await run_evaluator_agent(
            requirements=requirements,
            candidate_id="candidate_001",
            mcp_client=mcp,
            llm_client=llm,
        )


@pytest.mark.asyncio
async def test_evaluator_rejects_met_claim_without_quote(requirements, chunks):
    evaluation = valid_evaluation()
    evaluation["claims"][0]["quote"] = ""

    llm = FakeLLMClient(json.dumps(evaluation))
    mcp = FakeMCPClient(chunks)

    with pytest.raises(ValueError):
        await run_evaluator_agent(
            requirements=requirements,
            candidate_id="candidate_001",
            mcp_client=mcp,
            llm_client=llm,
        )


async def test_evaluator_accepts_fastmcp_dataclass_chunks(requirements, chunks):
    """Regression: FastMCP returns generated dataclass ``Root`` objects,
    which have attributes but no ``model_dump``. The Evaluator used to
    reject them with "Invalid candidate chunk returned by MCP"."""
    from dataclasses import make_dataclass

    Root = make_dataclass(
        "Root",
        [("chunk_id", str), ("candidate_id", str), ("section", str), ("text", str)],
    )
    llm = FakeLLMClient(json.dumps(valid_evaluation()))
    mcp = FakeMCPClient([Root(**chunk) for chunk in chunks])

    result = await run_evaluator_agent(
        requirements=requirements,
        candidate_id="candidate_001",
        mcp_client=mcp,
        llm_client=llm,
    )

    assert result.candidate_id == "candidate_001"
    assert "chunk_001" in llm.calls[0]["user"]


async def test_evaluator_rejects_malformed_chunk(requirements):
    llm = FakeLLMClient(json.dumps(valid_evaluation()))
    mcp = FakeMCPClient([{"chunk_id": "chunk_001"}])

    with pytest.raises(ValueError, match="Invalid candidate chunk"):
        await run_evaluator_agent(
            requirements=requirements,
            candidate_id="candidate_001",
            mcp_client=mcp,
            llm_client=llm,
        )


def test_parse_model_response_accepts_prose_around_json():
    response = (
        "Evaluation complete.\n```json\n"
        '{"candidate_id": "candidate_001", "claims": []}\n```\nDone.'
    )

    assert parse_model_response(response)["candidate_id"] == "candidate_001"
