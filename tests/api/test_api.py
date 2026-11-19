"""Tests for the FastAPI application and API routes."""

from datetime import datetime, timezone

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from ats.main import create_app
from ats.mcp_server import Runtime
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
    now = datetime.now(timezone.utc)

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
