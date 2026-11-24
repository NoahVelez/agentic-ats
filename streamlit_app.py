"""Streamlit recruiter dashboard for the Agentic ATS."""

import os

import httpx
import streamlit as st


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

API_KEY = os.getenv("APP_API_KEY", "")

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


st.set_page_config(
    page_title="Agentic ATS",
    page_icon="🎯",
    layout="wide",
)


def api_request(method, endpoint, **kwargs):
    """Send a request to the FastAPI backend."""
    try:
        response = httpx.request(
            method,
            f"{API_BASE_URL}{endpoint}",
            headers=HEADERS,
            timeout=30.0,
            **kwargs,
        )

        response.raise_for_status()

        if response.status_code == 204:
            return None

        return response.json()

    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        st.error(
            f"API error ({exc.response.status_code}): {detail}"
        )
    except httpx.RequestError as exc:
        st.error(f"Could not connect to the API: {exc}")

    return None


def show_candidate(candidate):
    """Display one candidate's evaluation."""
    with st.expander(
        f"#{candidate['rank']} — {candidate['candidate_id']} "
        f"(Score: {candidate['score']:.2f})"
    ):
        st.write("**Flags**")

        if candidate.get("flags"):
            for flag in candidate["flags"]:
                st.warning(flag)
        else:
            st.write("No flags.")

        evaluation = candidate.get("evaluation", {})

        st.write("**Requirement evaluations**")

        for claim in evaluation.get("claims", []):
            verdict = claim.get("verdict", "unknown")

            st.markdown(
                f"**{claim.get('requirement_id', 'Requirement')}** "
                f"— `{verdict}`"
            )

            st.write(claim.get("reasoning", ""))

            if claim.get("quote"):
                st.caption(f'Evidence: "{claim["quote"]}"')

            if claim.get("chunk_id"):
                st.caption(f"Evidence reference: {claim['chunk_id']}")

            st.divider()

        decision = candidate.get("decision")

        if decision:
            st.write(f"**Decision:** {decision.get('action')}")
            if decision.get("note"):
                st.write(decision["note"])


def show_run(run):
    """Display the current screening run."""
    st.subheader("Screening progress")

    col1, col2, col3 = st.columns(3)

    col1.metric("Status", run.get("status", "Unknown"))
    col2.metric("Current stage", run.get("stage", "Unknown"))
    col3.metric(
        "Candidates",
        len(run.get("shortlist", [])),
    )

    st.caption(f"Run ID: {run.get('run_id')}")

    if run.get("error"):
        st.error(run["error"])

    status = run.get("status")

    if status == "awaiting_clarification":
        st.warning("Recruiter clarification is required.")

        questions = run.get("pending_questions", [])

        if questions:
            with st.form("clarification_form"):
                answers = {}

                for question in questions:
                    label = question["text"]

                    if question.get("about_requirement"):
                        label += (
                            f"\nRelated requirement: "
                            f"{question['about_requirement']}"
                        )

                    answers[question["id"]] = st.text_area(
                        label,
                        key=f"answer_{run['run_id']}_{question['id']}",
                    )

                submitted = st.form_submit_button(
                    "Submit answers and resume screening"
                )

                if submitted:
                    if any(not value.strip() for value in answers.values()):
                        st.error("Please answer every question.")
                    else:
                        result = api_request(
                            "POST",
                            f"/api/runs/{run['run_id']}/clarifications",
                            json={"answers": answers},
                        )

                        if result:
                            st.success(
                                "Answers submitted. Screening has been queued."
                            )
                            st.rerun()

    elif status in {
        "queued",
        "requirements",
        "searching",
        "evaluating",
        "verifying",
    }:
        st.info(
            "Screening is in progress. Select Refresh to check the latest status."
        )

    elif status in {"awaiting_review", "completed"}:
        st.success("Screening results are available.")

    if run.get("shortlist"):
        st.subheader("Ranked candidates")

        for candidate in run["shortlist"]:
            show_candidate(candidate)

    usage = run.get("usage", {})

    with st.expander("Run usage"):
        st.write(f"Input tokens: {usage.get('tokens_in', 0):,}")
        st.write(f"Output tokens: {usage.get('tokens_out', 0):,}")
        st.write(f"Estimated cost: ${usage.get('cost_usd', 0):.4f}")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

st.title("🎯 Agentic ATS")
st.caption("AI-assisted recruiter screening dashboard")

health = api_request("GET", "/api/health")

if health:
    st.sidebar.success("Backend connected")
    st.sidebar.write(
        f"Vector store loaded: {health.get('vector_store_loaded')}"
    )
else:
    st.sidebar.error("Backend unavailable")

st.sidebar.caption(f"API: {API_BASE_URL}")

st.header("Job screening")

jobs = api_request("GET", "/api/jobs")

if jobs is None:
    st.error("Unable to load jobs. Check the backend and API key.")
    st.stop()

if not jobs:
    st.info("No jobs are currently available.")
    st.stop()

job_options = {
    f"{job['title']} ({job['job_id']})": job["job_id"]
    for job in jobs
}

selected_job_label = st.selectbox(
    "Select a job",
    options=list(job_options.keys()),
)

selected_job_id = job_options[selected_job_label]

job = api_request("GET", f"/api/jobs/{selected_job_id}")

if job:
    with st.expander("View job description"):
        st.write(job["text"])

if st.button(
    "Start candidate screening",
    type="primary",
):
    result = api_request(
        "POST",
        "/api/runs",
        json={
            "job_id": selected_job_id,
            "mode": "agentic",
        },
    )

    if result:
        st.session_state["active_run_id"] = result["run_id"]
        st.success(f"Screening started: {result['run_id']}")
        st.rerun()

st.divider()

st.header("Screening run")

run_id = st.text_input(
    "Run ID",
    value=st.session_state.get("active_run_id", ""),
    placeholder="Enter a screening run ID",
)

if st.button("Refresh run status"):
    if not run_id.strip():
        st.warning("Enter a run ID first.")
    else:
        run = api_request("GET", f"/api/runs/{run_id.strip()}")

        if run:
            st.session_state["active_run_id"] = run_id.strip()
            st.session_state["current_run"] = run

if st.session_state.get("current_run"):
    current_run = st.session_state["current_run"]

    if current_run.get("run_id") == run_id.strip():
        show_run(current_run)
    else:
        st.info("Select Refresh run status to load this run.")
elif st.session_state.get("active_run_id"):
    st.info("Select Refresh run status to retrieve your screening results.")
else:
    st.info("Start a screening run to see its progress and results.")
