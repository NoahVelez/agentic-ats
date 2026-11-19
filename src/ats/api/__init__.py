
"""FastAPI routes for the Agentic ATS run lifecycle."""

from __future__ import annotations

import hmac
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastmcp import Client
from pydantic import BaseModel

from ats.llm import create_llm_client
from ats.models import (
    ClarificationRequest,
    JobDescription,
    JobSummary,
    RunState,
)
from ats.orchestrator import run_screening_pipeline

if TYPE_CHECKING:
    from ats.mcp_server import Runtime


class CreateRunRequest(BaseModel):
    job_id: str
    mode: Literal["agentic", "keyword", "embedding"] = "agentic"


class ClarificationAnswersRequest(BaseModel):
    answers: dict[str, str]


def create_router(runtime: Runtime, server: Any) -> APIRouter:
    router = APIRouter()

    def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        configured_key = runtime.settings.app_api_key

        if configured_key is None:
            return

        expected = configured_key.get_secret_value()

        if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key.",
            )

    @router.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "vector_store_loaded": runtime.store_ready(),
        }

    # --------------------------------------------------
    # JOB ENDPOINTS
    # --------------------------------------------------

    @router.get(
        "/jobs",
        response_model=list[JobSummary],
        dependencies=[Depends(require_api_key)],
    )
    def list_jobs() -> list[JobSummary]:
        return [
            JobSummary(job_id=job.job_id, title=job.title)
            for job in runtime.jobs.values()
        ]

    @router.get(
        "/jobs/{job_id}",
        response_model=JobDescription,
        dependencies=[Depends(require_api_key)],
    )
    def get_job(job_id: str) -> JobDescription:
        job = runtime.jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        return job

    # --------------------------------------------------
    # BACKGROUND SCREENING RUN
    # --------------------------------------------------

    async def execute_run(
        run_id: str,
        recruiter_answers: dict[str, str] | None = None,
    ) -> None:
        state = runtime.run_store.get(run_id)

        state.status = "requirements"
        state.stage = "requirements"
        state.updated_at = datetime.now(timezone.utc)

        if recruiter_answers:
            state.recruiter_answers.update(recruiter_answers)

        runtime.run_store.save(state)

        try:
            llm_client = create_llm_client(runtime.settings)

            async with Client(server) as mcp_client:
                result = await run_screening_pipeline(
                    state.job_id,
                    mcp_client,
                    recruiter_answers=state.recruiter_answers or None,
                    llm_client=llm_client,
                    settings=runtime.settings,
                )

            state.updated_at = datetime.now(timezone.utc)

            if isinstance(result, ClarificationRequest):
                state.status = "awaiting_clarification"
                state.stage = "awaiting_clarification"
                state.pending_questions = result.questions

            else:
                _, _, _, shortlist = result

                state.status = "awaiting_review"
                state.stage = "awaiting_review"
                state.shortlist = shortlist
                state.pending_questions = []

        except Exception:
            state.status = "failed"
            state.stage = "failed"
            state.error = (
                "Screening run failed. Check the application logs."
            )
            state.updated_at = datetime.now(timezone.utc)

        runtime.run_store.save(state)

    # --------------------------------------------------
    # CREATE SCREENING RUN
    # --------------------------------------------------

    @router.post(
        "/runs",
        status_code=202,
        dependencies=[Depends(require_api_key)],
    )
    async def create_run(
        request: CreateRunRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, str]:

        if request.mode != "agentic":
            raise HTTPException(
                status_code=501,
                detail="Keyword and embedding modes are not implemented yet.",
            )

        if request.job_id not in runtime.jobs:
            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        now = datetime.now(timezone.utc)
        run_id = f"run_{uuid4().hex}"

        state = RunState(
            run_id=run_id,
            job_id=request.job_id,
            mode=request.mode,
            status="queued",
            stage="queued",
            created_at=now,
            updated_at=now,
        )

        runtime.run_store.create(state)
        background_tasks.add_task(execute_run, run_id)

        return {"run_id": run_id}

    # --------------------------------------------------
    # SUBMIT CLARIFICATION ANSWERS
    # --------------------------------------------------

    @router.post(
        "/runs/{run_id}/clarifications",
        status_code=202,
        dependencies=[Depends(require_api_key)],
    )
    async def submit_clarifications(
        run_id: str,
        request: ClarificationAnswersRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, str]:

        try:
            state = runtime.run_store.get(run_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid run ID.",
            ) from None
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Run not found.",
            ) from None

        if state.status != "awaiting_clarification":
            raise HTTPException(
                status_code=409,
                detail="This run is not awaiting clarification.",
            )

        answers = request.answers

        if not answers:
            raise HTTPException(
                status_code=422,
                detail="At least one answer is required.",
            )

        if any(not answer.strip() for answer in answers.values()):
            raise HTTPException(
                status_code=422,
                detail="Answers cannot be empty.",
            )

        expected_ids = {
            question.id for question in state.pending_questions
        }
        submitted_ids = set(answers)

        if submitted_ids != expected_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Submit an answer for every pending question.",
                    "missing_question_ids": sorted(expected_ids - submitted_ids),
                    "unexpected_question_ids": sorted(submitted_ids - expected_ids),
                },
            )

        state.recruiter_answers.update(
            {
                question_id: answer.strip()
                for question_id, answer in answers.items()
            }
        )

        state.pending_questions = []
        state.status = "queued"
        state.stage = "queued"
        state.updated_at = datetime.now(timezone.utc)

        runtime.run_store.save(state)

        background_tasks.add_task(
            execute_run,
            run_id,
            state.recruiter_answers,
        )

        return {
            "run_id": run_id,
            "status": "queued",
            "message": "Clarification answers saved. Screening will resume.",
        }

    # --------------------------------------------------
    # GET SCREENING RUN STATUS
    # --------------------------------------------------

    @router.get(
        "/runs/{run_id}",
        dependencies=[Depends(require_api_key)],
    )
    def get_run(run_id: str) -> RunState:
        try:
            return runtime.run_store.get(run_id)

        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid run ID.",
            ) from None

        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Run not found.",
            ) from None

    return router
