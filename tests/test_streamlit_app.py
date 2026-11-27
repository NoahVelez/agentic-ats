"""Headless tests for the Streamlit dashboard (streamlit.testing.v1.AppTest).

The real script runs; only ``httpx.request`` is replaced with a fake backend.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
RUN_ID = "run_" + "d" * 32
JOB = {
    "job_id": "jd_analytics_mgr",
    "title": "Analytics Manager",
    "text": "Lead a team of analysts. SQL required.",
}


def run_state(status, **extra):
    return {
        "run_id": RUN_ID,
        "job_id": JOB["job_id"],
        "mode": "agentic",
        "status": status,
        "stage": status,
        "pending_questions": [],
        "recruiter_answers": {},
        "shortlist": [],
        "usage": {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0},
        "error": None,
        **extra,
    }


class FakeBackend:
    def __init__(self, run=None, healthy=True):
        self.run = run or run_state("queued")
        self.healthy = healthy
        self.calls = []

    def __call__(self, method, url, **kwargs):
        path = url.split("://", 1)[1].split("/", 1)[1]
        path = "/" + path
        self.calls.append((method, path, kwargs.get("json")))
        request = httpx.Request(method, url)

        if not self.healthy:
            raise httpx.ConnectError("connection refused", request=request)

        def reply(status, body):
            return httpx.Response(status, json=body, request=request)

        if (method, path) == ("GET", "/api/health"):
            return reply(200, {"status": "ok", "vector_store_loaded": True})
        if (method, path) == ("GET", "/api/jobs"):
            return reply(200, [{"job_id": JOB["job_id"], "title": JOB["title"]}])
        if (method, path) == ("GET", f"/api/jobs/{JOB['job_id']}"):
            return reply(200, JOB)
        if (method, path) == ("POST", "/api/runs"):
            return reply(202, {"run_id": RUN_ID})
        if (method, path) == ("GET", f"/api/runs/{RUN_ID}"):
            return reply(200, self.run)
        if (method, path) == ("POST", f"/api/runs/{RUN_ID}/clarifications"):
            self.run = run_state("queued", recruiter_answers=kwargs["json"]["answers"])
            return reply(202, {"run_id": RUN_ID, "status": "queued"})
        if method == "GET" and path.startswith("/api/runs/"):
            return reply(404, {"detail": "Run not found."})
        return reply(404, {"detail": "Not Found"})


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(httpx, "request", fake)
    return fake


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_dashboard_shows_backend_status_and_jobs(backend):
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not app.exception
    assert [s.value for s in app.sidebar.success] == ["Backend connected"]
    assert any("Vector store loaded: True" in m.value for m in app.sidebar.markdown)
    assert app.selectbox[0].options == ["Analytics Manager (jd_analytics_mgr)"]
    assert any(JOB["text"] in m.value for m in app.markdown)


def test_backend_unavailable_shows_errors(monkeypatch):
    monkeypatch.setattr(httpx, "request", FakeBackend(healthy=False))
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not app.exception
    assert [e.value for e in app.sidebar.error] == ["Backend unavailable"]
    assert any("Unable to load jobs" in e.value for e in app.error)


def test_start_screening_then_refresh_shows_run(backend):
    backend.run = run_state(
        "awaiting_review",
        shortlist=[
            {
                "rank": 1,
                "candidate_id": "cand_0048",
                "score": 12.5,
                "flags": [],
                "evaluation": {
                    "candidate_id": "cand_0048",
                    "claims": [
                        {
                            "requirement_id": "R1",
                            "verdict": "met",
                            "quote": "Managed 8 analysts",
                            "chunk_id": "cand_0048:experience:1",
                            "reasoning": "Direct evidence.",
                        }
                    ],
                },
            }
        ],
        usage={"tokens_in": 58192, "tokens_out": 39661, "cost_usd": 0.0},
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    button(app, "Start candidate screening").click().run()

    assert ("POST", "/api/runs", {"job_id": JOB["job_id"], "mode": "agentic"}) in (backend.calls)
    assert app.text_input[0].value == RUN_ID

    button(app, "Refresh run status").click().run()

    assert not app.exception
    assert ("GET", f"/api/runs/{RUN_ID}", None) in backend.calls
    metrics = {m.label: m.value for m in app.metric}
    assert metrics == {
        "Status": "awaiting_review",
        "Current stage": "awaiting_review",
        "Candidates": "1",
    }
    assert any(RUN_ID in c.value for c in app.caption)
    assert any("Input tokens: 58,192" in m.value for m in app.markdown)
    assert any("Output tokens: 39,661" in m.value for m in app.markdown)


def test_refresh_unknown_run_shows_api_error(backend):
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.text_input[0].set_value("run_unknown").run()
    button(app, "Refresh run status").click().run()

    assert not app.exception
    assert any(e.value == "API error (404): Run not found." for e in app.error)
    assert not app.metric


def test_clarification_answers_are_submitted_and_form_clears(backend):
    backend.run = run_state(
        "awaiting_clarification",
        pending_questions=[
            {"id": "Q1", "text": "How many years?", "about_requirement": "R1"},
            {"id": "Q2", "text": "Is SQL mandatory?", "about_requirement": None},
        ],
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.text_input[0].set_value(RUN_ID).run()
    button(app, "Refresh run status").click().run()

    assert len(app.text_area) == 2

    # An empty answer is rejected without calling the API.
    app.text_area[0].set_value("5 years")
    button(app, "Submit answers and resume screening").click().run()
    assert any("Please answer every question." in e.value for e in app.error)
    assert not any(path.endswith("/clarifications") for _, path, _ in backend.calls)

    app.text_area[0].set_value("5 years")
    app.text_area[1].set_value("Yes")
    button(app, "Submit answers and resume screening").click().run()

    assert not app.exception
    assert (
        "POST",
        f"/api/runs/{RUN_ID}/clarifications",
        {"answers": {"Q1": "5 years", "Q2": "Yes"}},
    ) in backend.calls
    # The refreshed (queued) run replaces the stale clarification form.
    assert len(app.text_area) == 0
    assert {m.label: m.value for m in app.metric}["Status"] == "queued"


def test_typed_run_id_is_kept_after_a_run_was_loaded(backend):
    """Regression: the Run ID box was rebuilt whenever the loaded run changed,
    so the next ID a recruiter typed was discarded and the old one came back."""
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.text_input[0].set_value(RUN_ID).run()
    button(app, "Refresh run status").click().run()
    assert app.metric

    app.text_input[0].set_value("run_unknown").run()
    assert app.text_input[0].value == "run_unknown"

    button(app, "Refresh run status").click().run()
    assert not app.exception
    assert app.text_input[0].value == "run_unknown"
    assert any("Run not found" in error.value for error in app.error)
