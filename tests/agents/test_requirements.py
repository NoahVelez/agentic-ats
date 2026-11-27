import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr, create_model

from ats.agents.requirements import (
    _get_job_from_mcp_result,
    build_user_prompt,
    load_system_prompt,
    parse_model_response,
    run_requirements_agent,
)
from ats.config import Settings
from ats.llm import OpenAICompatibleClient, create_llm_client
from ats.models import ClarificationRequest, JobDescription, RequirementSet


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
        text=("Required: SQL and Python experience. Power BI experience is preferred."),
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
    assert mcp_client.calls == [("get_job", {"job_id": "JOB-001"})]


@pytest.mark.asyncio
async def test_requirements_agent_returns_clarification_request(mcp_client, settings):
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
async def test_requirements_agent_rejects_non_sequential_ids(mcp_client, settings):
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
async def test_requirements_agent_rejects_weight_above_three(mcp_client, settings):
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


def test_create_claude_client(monkeypatch):
    captured = {}

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("ats.llm.AsyncAnthropic", FakeAsyncAnthropic)

    settings = Settings(
        llm_provider="claude",
        llm_model="claude-sonnet-4-5",
        llm_api_key=SecretStr("test-key"),
    )

    client = create_llm_client(settings)

    from ats.llm import AnthropicClient

    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-4-5"
    assert captured["api_key"] == "test-key"


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


def test_get_job_from_mcp_result_unwraps_pydantic_root(job):
    Root = create_model(
        "Root",
        job_id=(str, ...),
        title=(str, ...),
        text=(str, ...),
    )
    root = Root(
        job_id=job.job_id,
        title=job.title,
        text=job.text,
    )
    result = SimpleNamespace(
        is_error=False,
        data=root,
        structured_content=None,
    )

    parsed = _get_job_from_mcp_result(result)

    assert parsed == job


# CLAUDE CLIENT


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_claude_client(monkeypatch, response):
    messages = FakeMessages(response)

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            self.messages = messages

    monkeypatch.setattr("ats.llm.AsyncAnthropic", FakeAsyncAnthropic)

    from ats.llm import AnthropicClient

    return AnthropicClient(model="claude-haiku-5-5", api_key="test-key"), messages


def claude_response(*blocks, stop_reason="end_turn", tokens_in=10, tokens_out=5):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


async def test_claude_complete_sends_supported_arguments_and_returns_text(
    monkeypatch,
):
    response = claude_response(
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text='{"ok": true}'),
    )
    client, messages = make_claude_client(monkeypatch, response)

    text = await client.complete("system prompt", "user prompt")

    assert text == '{"ok": true}'
    call = messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["system"] == "system prompt"
    assert call["messages"] == [{"role": "user", "content": "user prompt"}]
    # Thinking tokens count toward max_tokens, so leave headroom.
    assert call["max_tokens"] >= 16000
    # The Anthropic API rejects sampling parameters on current models.
    assert "temperature" not in call
    assert "top_p" not in call


async def test_claude_complete_accumulates_token_usage(monkeypatch):
    response = claude_response(
        SimpleNamespace(type="text", text="{}"),
        tokens_in=120,
        tokens_out=30,
    )
    client, _ = make_claude_client(monkeypatch, response)

    await client.complete("s", "u")
    await client.complete("s", "u")

    assert client.tokens_in == 240
    assert client.tokens_out == 60
    # Claude Haiku 5.5: $0.10 in / $0.50 out per million tokens.
    assert client.cost_usd == pytest.approx((240 * 0.10 + 60 * 0.50) / 1_000_000)


def test_estimate_claude_cost_uses_long_prompt_rate_and_ignores_unknown_models():
    from ats.llm import estimate_claude_cost

    # Exactly 100K input tokens still bills at the standard rate.
    assert estimate_claude_cost("claude-haiku-5-5", 100_000, 1_000_000) == pytest.approx(0.51)
    # Haiku 5.5 prompts over 100K input tokens bill at $0.50 / $2.50.
    assert estimate_claude_cost("claude-haiku-5-5", 200_000, 10_000) == (
        pytest.approx((200_000 * 0.50 + 10_000 * 2.50) / 1_000_000)
    )
    assert estimate_claude_cost("claude-opus-5-5", 1_000, 1_000) == pytest.approx(0.024)
    assert estimate_claude_cost("unpriced-model", 1_000, 1_000) == 0.0


async def test_claude_complete_rejects_truncated_response(monkeypatch):
    response = claude_response(
        SimpleNamespace(type="text", text='{"requirements": ['),
        stop_reason="max_tokens",
    )
    client, _ = make_claude_client(monkeypatch, response)

    with pytest.raises(ValueError, match="truncated"):
        await client.complete("s", "u")

    # Tokens were still spent and must still be counted.
    assert client.tokens_out == 5


async def test_claude_complete_rejects_refusal(monkeypatch):
    response = claude_response(stop_reason="refusal")
    client, _ = make_claude_client(monkeypatch, response)

    with pytest.raises(ValueError, match="declined"):
        await client.complete("s", "u")


# CLAUDE OUTPUT AND FASTMCP RESULT SHAPES


def test_extract_json_text_handles_claude_wrappers():
    from ats.llm import extract_json_text

    assert extract_json_text('{"a": 1}') == '{"a": 1}'
    assert extract_json_text('Here you go:\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_text("no json here") == "no json here"


def test_parse_model_response_accepts_prose_before_fenced_json():
    response = (
        "Here are the structured requirements:\n\n"
        '```json\n{"job_id": "job_001", "requirements": []}\n```'
    )

    assert parse_model_response(response)["job_id"] == "job_001"


def test_parse_model_response_still_rejects_json_array():
    with pytest.raises(ValueError, match="JSON object"):
        parse_model_response('[{"job_id": "job_001"}]')


def test_get_job_from_mcp_result_accepts_fastmcp_dataclass_root(job):
    """FastMCP builds result.data as a generated *dataclass* named Root."""
    from dataclasses import make_dataclass

    Root = make_dataclass("Root", [("job_id", str), ("title", str), ("text", str)])
    result = SimpleNamespace(
        is_error=False,
        data=Root(job_id=job.job_id, title=job.title, text=job.text),
        structured_content=None,
    )

    assert _get_job_from_mcp_result(result) == job


# CLARIFICATION WORKFLOW

VALID_REQUIREMENTS = {
    "requirements": [
        {
            "id": "R1",
            "text": "SQL experience",
            "kind": "must",
            "weight": 3,
            "equivalents": [],
            "assumption": None,
        }
    ]
}


async def test_clarification_questions_get_stable_sequential_ids(mcp_client, settings):
    response = {
        "questions": [
            {"id": "q-sql", "text": "How many years of SQL?"},
            {"id": "q-sql", "text": "Is Python required?", "about_requirement": "R2"},
            {"text": "Is Power BI mandatory?"},
            {"id": "Q9", "text": "Over the limit and should be dropped."},
        ]
    }

    result = await run_requirements_agent(
        "JOB-001",
        mcp_client,
        llm_client=FakeLLMClient(json.dumps(response)),
        settings=settings,
    )

    assert isinstance(result, ClarificationRequest)
    assert [q.id for q in result.questions] == ["Q1", "Q2", "Q3"]
    assert result.questions[1].about_requirement == "R2"


async def test_empty_questions_list_does_not_trigger_clarification(mcp_client, settings):
    """Regression: {"requirements": [...], "questions": []} used to become a
    ClarificationRequest with zero questions, stranding the run."""
    response = {**VALID_REQUIREMENTS, "questions": []}

    result = await run_requirements_agent(
        "JOB-001",
        mcp_client,
        llm_client=FakeLLMClient(json.dumps(response)),
        settings=settings,
    )

    assert isinstance(result, RequirementSet)
    assert result.requirements[0].id == "R1"


async def test_complete_requirements_do_not_trigger_clarification(mcp_client, settings):
    result = await run_requirements_agent(
        "JOB-001",
        mcp_client,
        llm_client=FakeLLMClient(json.dumps(VALID_REQUIREMENTS)),
        settings=settings,
    )

    assert isinstance(result, RequirementSet)


async def test_recruiter_answers_are_sent_and_requirements_returned(mcp_client, settings):
    llm = FakeLLMClient(json.dumps(VALID_REQUIREMENTS))

    result = await run_requirements_agent(
        "JOB-001",
        mcp_client,
        recruiter_answers={"Q1": "At least 3 years"},
        llm_client=llm,
        settings=settings,
    )

    assert isinstance(result, RequirementSet)
    assert "At least 3 years" in llm.last_user_prompt
    assert "do not ask further questions" in llm.last_user_prompt


async def test_repeat_clarification_after_answers_is_rejected(mcp_client, settings):
    response = {"questions": [{"id": "Q1", "text": "Asked again?"}]}

    with pytest.raises(ValueError, match="clarification again"):
        await run_requirements_agent(
            "JOB-001",
            mcp_client,
            recruiter_answers={"Q1": "At least 3 years"},
            llm_client=FakeLLMClient(json.dumps(response)),
            settings=settings,
        )


async def test_invalid_clarification_questions_are_rejected(mcp_client, settings):
    for questions in ("not a list", [{"id": "Q1"}], [{"text": "   "}], ["bare"]):
        with pytest.raises(ValueError):
            await run_requirements_agent(
                "JOB-001",
                mcp_client,
                llm_client=FakeLLMClient(json.dumps({"questions": questions})),
                settings=settings,
            )
