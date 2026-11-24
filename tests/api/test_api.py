"""Tests for the FastAPI application and API routes."""

import asyncio
import json
from datetime import UTC, datetime

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from ats.main import create_app
from ats.mcp_server import Runtime, create_mcp
from ats.models import Question, RunState
from tests.conftest import JOB_ID
from tests.mcp.conftest import make_settings


def make_runtime(tmp_path, **overrides):
    settings_kwargs = {
        "data_dir": tmp_path / "jobs",
        "chroma_dir": tmp_path / "chroma",
        "runs_dir": tmp_path / "runs",
    }
    settings_kwargs.update(overrides)

    settings = make_settings(**settings_kwargs)
    return Runtime(settings)


# HEALTH ENDPOINT TESTS

def test_health_endpoint(tmp_path):
    runtime = make_runtime(tmp_path)
    app = create_app(runtime)
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint_does_not_require_api_key(tmp_path):
    runtime = make_runtime(tmp_path, app_api_key="test-secret")
    app = create_app(runtime)
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# API AUTHENTICATION TESTS

def test_protected_endpoint_rejects_missing_api_key(tmp_path):
    runtime = make_runtime(tmp_path, app_api_key="test-secret")
    app = create_app(runtime)
    client = TestClient(app)

    response = client.get("/api/runs/run_does_not_exist")

    assert response.status_code == 401


def test_protected_endpoint_rejects_incorrect_api_key(tmp_path):
    runtime = make_runtime(tmp_path, app_api_key="test-secret")
    app = create_app(runtime)
    client = TestClient(app)

    response = client.get(
        "/api/runs/run_does_not_exist",
        headers={"X-API-Key": "wrong-secret"},
    )

    assert response.status_code == 401


# CREATE RUN TEST

def test_create_run_returns_202_and_saves_initial_state(
    tmp_path,
    dataset,
    monkeypatch,
):
    monkeypatch.setattr(
        BackgroundTasks,
        "add_task",
        lambda self, func, *args, **kwargs: None,
    )

    runtime = make_runtime(tmp_path, data_dir=dataset)
    app = create_app(runtime)
    client = TestClient(app)

    response = client.post("/api/runs", json={"job_id": JOB_ID})

    assert response.status_code == 202

    run_id = response.json()["run_id"]
    state = runtime.run_store.get(run_id)

    assert state.run_id == run_id
    assert state.job_id == JOB_ID
    assert state.status == "queued"
    assert state.stage == "queued"


# JOB ENDPOINT TESTS

def test_list_jobs_returns_job_summaries(tmp_path, dataset):
    runtime = make_runtime(tmp_path, data_dir=dataset)
    app = create_app(runtime)
    client = TestClient(app)

    response = client.get("/api/jobs")

    assert response.status_code == 200
    assert len(response.json()) > 0
    assert "job_id" in response.json()[0]
    assert "title" in response.json()[0]


def test_get_job_returns_full_description(tmp_path, dataset):
    runtime = make_runtime(tmp_path, data_dir=dataset)
    app = create_app(runtime)
    client = TestClient(app)

    job_id = next(iter(runtime.jobs))
    response = client.get(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    assert "title" in response.json()
    assert "text" in response.json()


def test_get_unknown_job_returns_404(tmp_path, dataset):
    runtime = make_runtime(tmp_path, data_dir=dataset)
    app = create_app(runtime)
    client = TestClient(app)

    response = client.get("/api/jobs/unknown_job")

    assert response.status_code == 404


def test_jobs_endpoint_requires_api_key(tmp_path, dataset):
    runtime = make_runtime(
        tmp_path,
        data_dir=dataset,
        app_api_key="test-secret",
    )
    app = create_app(runtime)
    client = TestClient(app)

    response = client.get("/api/jobs")

    assert response.status_code == 401


# CLARIFICATION TEST HELPER

def create_clarification_run(runtime, status="awaiting_clarification"):
    now = datetime.now(UTC)

    pending_questions = []

    if status == "awaiting_clarification":
        pending_questions = [
            Question(
                id="Q1",
                text="How many years of experience are required?",
            ),
            Question(
                id="Q2",
                text="Is a bachelor's degree mandatory?",
            ),
        ]

    state = RunState(
        run_id="run_" + "a" * 32,
        job_id=JOB_ID,
        mode="agentic",
        status=status,
        stage=status,
        pending_questions=pending_questions,
        created_at=now,
        updated_at=now,
    )

    runtime.run_store.create(state)
    return state


# CLARIFICATION ENDPOINT TESTS

def test_submit_clarifications_saves_answers_and_queues_run(
    tmp_path,
    dataset,
    monkeypatch,
):
    scheduled_tasks = []

    monkeypatch.setattr(
        BackgroundTasks,
        "add_task",
        lambda self, func, *args, **kwargs: scheduled_tasks.append(
            (func, args, kwargs)
        ),
    )

    runtime = make_runtime(tmp_path, data_dir=dataset)
    state = create_clarification_run(runtime)

    app = create_app(runtime)
    client = TestClient(app)

    response = client.post(
        f"/api/runs/{state.run_id}/clarifications",
        json={
            "answers": {
                "Q1": "At least 3 years",
                "Q2": "Preferred, not mandatory",
            }
        },
    )

    assert response.status_code == 202

    saved_state = runtime.run_store.get(state.run_id)

    assert saved_state.status == "queued"
    assert saved_state.stage == "queued"
    assert saved_state.pending_questions == []
    assert saved_state.recruiter_answers == {
        "Q1": "At least 3 years",
        "Q2": "Preferred, not mandatory",
    }

    assert len(scheduled_tasks) == 1
    assert scheduled_tasks[0][1][0] == state.run_id
    assert scheduled_tasks[0][1][1] == saved_state.recruiter_answers


def test_submit_clarifications_rejects_missing_answers(
    tmp_path,
    dataset,
    monkeypatch,
):
    monkeypatch.setattr(
        BackgroundTasks,
        "add_task",
        lambda self, func, *args, **kwargs: None,
    )

    runtime = make_runtime(tmp_path, data_dir=dataset)
    state = create_clarification_run(runtime)

    app = create_app(runtime)
    client = TestClient(app)

    response = client.post(
        f"/api/runs/{state.run_id}/clarifications",
        json={"answers": {"Q1": "At least 3 years"}},
    )

    assert response.status_code == 422

    saved_state = runtime.run_store.get(state.run_id)

    assert saved_state.status == "awaiting_clarification"
    assert len(saved_state.pending_questions) == 2


def test_submit_clarifications_rejects_run_not_awaiting_clarification(
    tmp_path,
    dataset,
):
    runtime = make_runtime(tmp_path, data_dir=dataset)
    state = create_clarification_run(runtime, status="queued")

    app = create_app(runtime)
    client = TestClient(app)

    response = client.post(
        f"/api/runs/{state.run_id}/clarifications",
        json={"answers": {"Q1": "At least 3 years"}},
    )

    assert response.status_code == 409


# END-TO-END SCREENING RUN (real FastMCP server, fake Claude)


class FakeClaudeClient:
    """Stands in for AnthropicClient: answers like Claude and counts tokens."""

    def __init__(self):
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0

    async def complete(self, system, user):
        self.calls += 1
        self.tokens_in += 100
        self.tokens_out += 20

        if "Requirements agent" in system:
            requirements = {
                "requirements": [
                    {
                        "id": "R1",
                        "text": "Experience managing a team of analysts",
                        "kind": "must",
                        "weight": 3,
                        "equivalents": ["led a team"],
                        "assumption": None,
                    },
                    {
                        "id": "R2",
                        "text": "Proficiency in SQL",
                        "kind": "nice",
                        "weight": 2,
                        "equivalents": [],
                        "assumption": None,
                    },
                ]
            }
            # Claude often adds a sentence before a fenced JSON block.
            return (
                "Here are the requirements:\n\n```json\n"
                + json.dumps(requirements)
                + "\n```"
            )

        candidate_id = user.split("Candidate ID: ", 1)[1].splitlines()[0].strip()

        return json.dumps(
            {
                "candidate_id": candidate_id,
                "claims": [
                    {
                        "requirement_id": requirement_id,
                        "verdict": "missing",
                        "quote": None,
                        "chunk_id": None,
                        "reasoning": "No evidence found.",
                    }
                    for requirement_id in ("R1", "R2")
                ],
            }
        )


def test_screening_run_completes_through_real_mcp_server(
    tmp_path,
    dataset,
    indexed,
    monkeypatch,
):
    """Regression: FastMCP returns dataclass ``Root`` objects (not dicts or
    Pydantic models). The Evaluator rejected them, so every live run failed."""
    scheduled_tasks = []

    monkeypatch.setattr(
        BackgroundTasks,
        "add_task",
        lambda self, func, *args, **kwargs: scheduled_tasks.append(
            (func, args, kwargs)
        ),
    )

    fake_llm = FakeClaudeClient()
    monkeypatch.setattr(
        "ats.api.create_llm_client",
        lambda settings: fake_llm,
    )

    runtime = make_runtime(
        tmp_path,
        data_dir=dataset,
        chroma_dir=indexed[0],
        strong_hit=0.3,
        max_pool_size=3,
    )
    app = create_app(runtime, create_mcp(runtime))
    client = TestClient(app)

    response = client.post("/api/runs", json={"job_id": JOB_ID})
    assert response.status_code == 202
    run_id = response.json()["run_id"]

    assert len(scheduled_tasks) == 1
    func, args, kwargs = scheduled_tasks[0]
    asyncio.run(func(*args, **kwargs))

    state = runtime.run_store.get(run_id)

    assert state.error is None
    assert state.status == "awaiting_review"
    assert 1 <= len(state.shortlist) <= 3
    assert state.shortlist[0].rank == 1
    assert state.usage.tokens_in == fake_llm.tokens_in > 0
    assert state.usage.tokens_out == fake_llm.tokens_out > 0


# REQUEST VALIDATION AND ERROR HANDLING


def test_create_run_validates_request_body_and_mode(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(
        BackgroundTasks,
        "add_task",
        lambda self, func, *args, **kwargs: None,
    )
    runtime = make_runtime(tmp_path, data_dir=dataset)
    client = TestClient(create_app(runtime))

    assert client.post("/api/runs", json={}).status_code == 422
    assert client.post("/api/runs", content="not json").status_code == 422
    assert (
        client.post("/api/runs", json={"job_id": JOB_ID, "mode": "bogus"}).status_code
        == 422
    )

    for mode in ("keyword", "embedding"):
        response = client.post("/api/runs", json={"job_id": JOB_ID, "mode": mode})
        assert response.status_code == 501

    response = client.post("/api/runs", json={"job_id": "jd_does_not_exist"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found."}


def test_get_run_returns_state_and_handles_bad_ids(tmp_path, dataset):
    runtime = make_runtime(tmp_path, data_dir=dataset)
    state = create_clarification_run(runtime)
    client = TestClient(create_app(runtime))

    response = client.get(f"/api/runs/{state.run_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "awaiting_clarification"
    assert [q["id"] for q in body["pending_questions"]] == ["Q1", "Q2"]

    assert client.get("/api/runs/-bad").status_code == 400
    assert client.get("/api/runs/not-a-known-run").status_code == 404
    assert client.get("/api/runs/run_" + "b" * 32).status_code == 404


def test_submit_clarifications_rejects_bad_runs_and_answers(tmp_path, dataset):
    runtime = make_runtime(tmp_path, data_dir=dataset)
    state = create_clarification_run(runtime)
    client = TestClient(create_app(runtime))
    url = f"/api/runs/{state.run_id}/clarifications"
    both = {"Q1": "3 years", "Q2": "No"}

    assert (
        client.post("/api/runs/-bad/clarifications", json={"answers": both}).status_code
        == 400
    )
    assert (
        client.post(
            "/api/runs/run_" + "c" * 32 + "/clarifications", json={"answers": both}
        ).status_code
        == 404
    )
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"answers": {}}).status_code == 422
    assert (
        client.post(url, json={"answers": {"Q1": "3 years", "Q2": "   "}}).status_code
        == 422
    )

    response = client.post(url, json={"answers": {**both, "Q7": "extra"}})
    assert response.status_code == 422
    assert response.json()["detail"]["unexpected_question_ids"] == ["Q7"]

    saved = runtime.run_store.get(state.run_id)
    assert saved.status == "awaiting_clarification"
    assert saved.recruiter_answers == {}


class ClarifyingFakeClaudeClient(FakeClaudeClient):
    """Asks one question on the first Requirements call, then answers."""

    def __init__(self):
        super().__init__()
        self.requirement_prompts = []

    async def complete(self, system, user):
        if "Requirements agent" in system:
            self.requirement_prompts.append(user)
            if len(self.requirement_prompts) == 1:
                self.calls += 1
                return json.dumps(
                    {
                        "questions": [
                            {
                                "id": "years",
                                "text": "How many years of management are required?",
                                "about_requirement": "R1",
                            }
                        ]
                    }
                )
        return await super().complete(system, user)


def test_clarification_round_trip_resumes_screening(
    tmp_path,
    dataset,
    indexed,
    monkeypatch,
):
    scheduled_tasks = []
    monkeypatch.setattr(
        BackgroundTasks,
        "add_task",
        lambda self, func, *args, **kwargs: scheduled_tasks.append(
            (func, args, kwargs)
        ),
    )
    fake_llm = ClarifyingFakeClaudeClient()
    monkeypatch.setattr("ats.api.create_llm_client", lambda settings: fake_llm)

    runtime = make_runtime(
        tmp_path,
        data_dir=dataset,
        chroma_dir=indexed[0],
        strong_hit=0.3,
        max_pool_size=3,
    )
    client = TestClient(create_app(runtime, create_mcp(runtime)))

    run_id = client.post("/api/runs", json={"job_id": JOB_ID}).json()["run_id"]
    func, args, kwargs = scheduled_tasks.pop()
    asyncio.run(func(*args, **kwargs))

    body = client.get(f"/api/runs/{run_id}").json()
    assert body["status"] == "awaiting_clarification"
    assert body["pending_questions"] == [
        {
            "id": "Q1",
            "text": "How many years of management are required?",
            "about_requirement": "R1",
        }
    ]

    response = client.post(
        f"/api/runs/{run_id}/clarifications",
        json={"answers": {"Q1": "At least 5 years"}},
    )
    assert response.status_code == 202
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "queued"

    func, args, kwargs = scheduled_tasks.pop()
    asyncio.run(func(*args, **kwargs))

    body = client.get(f"/api/runs/{run_id}").json()
    assert body["error"] is None
    assert body["status"] == "awaiting_review"
    assert body["pending_questions"] == []
    assert body["recruiter_answers"] == {"Q1": "At least 5 years"}
    assert body["shortlist"]
    assert "At least 5 years" in fake_llm.requirement_prompts[-1]
    assert body["usage"]["tokens_in"] == fake_llm.tokens_in
