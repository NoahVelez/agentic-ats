
import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from ats.agents.requirements import (
    build_user_prompt,
    load_system_prompt,
    parse_model_response,
    run_requirements_agent,
)
from ats.config import Settings
from ats.llm import OpenAICompatibleClient, create_llm_client
from ats.models import JobDescription, RequirementSet, ClarificationRequest


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.last_system_prompt = None
        self.last_user_prompt = None

    async def complete(self, system, user):
        self.last_system_prompt = system
        self.last_user_prompt = user
        return self.response


class FakeMCPClient:
    def __init__(self, job):
        self.job = job
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            is_error=False,
            data=self.job,
            structured_content=None,
        )


@pytest.fixture
def job():
    return JobDescription(
        job_id="JOB-001",
        title="Data Analyst",
        text=(
            "Required: SQL and Python experience. "
            "Power BI experience is preferred."
        ),
    )


@pytest.fixture
def settings():
    return SimpleNamespace(max_clarify_questions=3)


@pytest.fixture
def mcp_client(job):
    return FakeMCPClient(job)


def test_load_system_prompt():
    prompt = load_system_prompt()
    assert "Requirements agent" in prompt
    assert "Do not invent requirements" in prompt


def test_build_user_prompt_includes_job_details(job):
    prompt = build_user_prompt(job, max_questions=3)
    assert job.job_id in prompt
    assert job.title in prompt
    assert job.text in prompt
    assert "Maximum clarification questions: 3" in prompt


def test_parse_model_response_valid_json():
    assert parse_model_response('{"status": "ok"}') == {"status": "ok"}


def test_parse_model_response_code_fence():
    result = parse_model_response('```json\n{"status": "ok"}\n```')
    assert result == {"status": "ok"}


def test_parse_model_response_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_model_response("not valid JSON")


def test_parse_model_response_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        parse_model_response('["one", "two"]')


@pytest.mark.asyncio
async def test_requirements_agent_returns_requirement_set(mcp_client, settings):
    response = {
        "requirements": [
            {
                "id": "R1",
                "text": "Demonstrated SQL experience",
                "kind": "must",
                "weight": 3,
                "equivalents": ["SQL querying"],
                "assumption": None,
            },
            {
                "id": "R2",
                "text": "Experience using Power BI",
                "kind": "nice",
                "weight": 2,
                "equivalents": ["Power BI dashboards"],
                "assumption": None,
            },
        ]
    }

    client = FakeLLMClient(json.dumps(response))

    result = await run_requirements_agent(
        "JOB-001",
        mcp_client,
        llm_client=client,
        settings=settings,
    )

    assert isinstance(result, RequirementSet)
    assert result.job_id == "JOB-001"
    assert len(result.requirements) == 2
    assert result.requirements[0].id == "R1"
    assert result.requirements[1].id == "R2"
    assert client.last_system_prompt
    assert "SQL" in client.last_user_prompt
    assert mcp_client.calls == [
        ("get_job", {"job_id": "JOB-001"})
    ]


@pytest.mark.asyncio
async def test_requirements_agent_returns_clarification_request(
    mcp_client, settings
):
    response = {
        "questions": [
            {
                "id": "Q1",
                "text": "How many years of SQL experience are required?",
                "about_requirement": "R1",
            }
        ]
    }

    client = FakeLLMClient(json.dumps(response))

    result = await run_requirements_agent(
        "JOB-001",
        mcp_client,
        llm_client=client,
        settings=settings,
    )

    assert isinstance(result, ClarificationRequest)
    assert len(result.questions) == 1
    assert result.questions[0].id == "Q1"


@pytest.mark.asyncio
async def test_requirements_agent_rejects_non_sequential_ids(
    mcp_client, settings
):
    response = {
        "requirements": [
            {
                "id": "R2",
                "text": "Demonstrated SQL experience",
                "kind": "must",
                "weight": 3,
                "equivalents": [],
                "assumption": None,
            }
        ]
    }

    client = FakeLLMClient(json.dumps(response))

    with pytest.raises(ValueError, match="sequential"):
        await run_requirements_agent(
            "JOB-001",
            mcp_client,
            llm_client=client,
            settings=settings,
        )


@pytest.mark.asyncio
async def test_requirements_agent_rejects_weight_above_three(
    mcp_client, settings
):
    response = {
        "requirements": [
            {
                "id": "R1",
                "text": "Demonstrated SQL experience",
                "kind": "must",
                "weight": 4,
                "equivalents": [],
                "assumption": None,
            }
        ]
    }

    client = FakeLLMClient(json.dumps(response))

    with pytest.raises(ValueError, match="invalid weight"):
        await run_requirements_agent(
            "JOB-001",
            mcp_client,
            llm_client=client,
            settings=settings,
        )


def test_create_groq_client(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("ats.llm.AsyncOpenAI", FakeAsyncOpenAI)

    settings = Settings(
        llm_provider="groq",
        llm_model="openai/gpt-oss-20b",
        llm_api_key=SecretStr("test-key"),
    )

    client = create_llm_client(settings)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "openai/gpt-oss-20b"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.groq.com/openai/v1"


def test_create_openai_client(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("ats.llm.AsyncOpenAI", FakeAsyncOpenAI)

    settings = Settings(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key=SecretStr("test-key"),
    )

    client = create_llm_client(settings)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "gpt-4o-mini"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] is None


def test_create_llm_client_rejects_missing_api_key():
    settings = Settings(
        llm_provider="groq",
        llm_model="openai/gpt-oss-20b",
    )

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        create_llm_client(settings)


def test_create_llm_client_rejects_unsupported_provider():
    settings = Settings(
        llm_provider="unsupported",
        llm_model="some-model",
        llm_api_key=SecretStr("test-key"),
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(settings)
