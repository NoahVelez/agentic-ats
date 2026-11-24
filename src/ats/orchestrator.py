
"""Plain-Python orchestration for the Agentic ATS workflow."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import Client

from ats.agents.evaluator import run_evaluator_agent
from ats.agents.requirements import run_requirements_agent
from ats.agents.search import run_search_agent
from ats.config import Settings, get_settings
from ats.llm import LLMClient
from ats.models import (
    CandidateEvaluation,
    CandidatePool,
    ClarificationRequest,
    RequirementSet,
    ShortlistEntry,
    TraceEvent,
)
from ats.tracing import append_event


def _extract_mcp_result(result: Any) -> Any:
    """Extract structured data from a FastMCP tool result."""
    if getattr(result, "is_error", False):
        raise RuntimeError("An MCP tool returned an error.")

    data = getattr(result, "data", None)

    if data is not None:
        return data

    structured = getattr(result, "structured_content", None)

    if isinstance(structured, dict):
        return structured.get("result", structured)

    raise RuntimeError("The MCP tool returned no usable data.")


def _write_trace(
    runs_dir: Path,
    run_id: str,
    *,
    stage: str,
    event: str,
    agent: str | None = None,
    round_number: int | None = None,
    tool: str | None = None,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    latency_ms: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Write one event using the project's existing trace format."""
    trace_event = TraceEvent(
        ts=datetime.now(UTC),
        run_id=run_id,
        stage=stage,
        agent=agent,
        event=event,
        round=round_number,
        tool=tool,
        input=input_data or {},
        output=output_data or {},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )

    append_event(runs_dir, trace_event)


def score_candidate(
    requirements: RequirementSet,
    evaluation: CandidateEvaluation,
) -> tuple[float, list[str]]:
    """Calculate weighted points from verified claims.

    Met = 1.0, partial = 0.5, missing = 0.
    Unverified claims do not contribute to the score.
    """
    requirement_by_id = {
        requirement.id: requirement
        for requirement in requirements.requirements
    }

    score = 0.0
    flags: list[str] = []

    for claim in evaluation.claims:
        requirement = requirement_by_id.get(claim.requirement_id)

        if requirement is None:
            continue

        if claim.verdict == "unverified":
            continue

        if claim.verdict == "met":
            score += requirement.weight

        elif claim.verdict == "partial":
            score += requirement.weight * 0.5

        if (
            requirement.kind == "must"
            and claim.verdict == "missing"
        ):
            flags.append("missing_must_have")

    return score, sorted(set(flags))
async def evaluate_candidates(
    requirements: RequirementSet,
    candidate_pool: CandidatePool,
    mcp_client: Client,
    *,
    llm_client: LLMClient | None = None,
    settings: Settings | None = None,
) -> list[CandidateEvaluation]:
    """Evaluate candidates concurrently within the configured limit."""
    settings = settings or get_settings()
    semaphore = asyncio.Semaphore(settings.eval_concurrency)

    async def evaluate_one(candidate_id: str) -> CandidateEvaluation:
        async with semaphore:
            return await run_evaluator_agent(
                requirements,
                candidate_id,
                mcp_client,
                llm_client=llm_client,
                settings=settings,
            )

    return await asyncio.gather(
        *(evaluate_one(candidate_id) for candidate_id in candidate_pool.candidate_ids)
    )
def rank_candidates(
    requirements: RequirementSet,
    evaluations: list[CandidateEvaluation],
) -> list[ShortlistEntry]:
    """Score and rank candidates, placing missing-must-have candidates last."""
    ranked = []

    for evaluation in evaluations:
        score, flags = score_candidate(requirements, evaluation)

        ranked.append({
            "candidate_id": evaluation.candidate_id,
            "score": score,
            "flags": flags,
            "evaluation": evaluation,
        })

    ranked.sort(
        key=lambda candidate: (
            "missing_must_have" in candidate["flags"],
            -candidate["score"],
            candidate["candidate_id"],
        )
    )

    return [
        ShortlistEntry(
            rank=index,
            candidate_id=candidate["candidate_id"],
            score=candidate["score"],
            flags=candidate["flags"],
            evaluation=candidate["evaluation"],
        )
        for index, candidate in enumerate(ranked, start=1)
    ]

async def prepare_screening(
    job_id: str,
    mcp_client: Client,
    *,
    recruiter_answers: dict[str, Any] | None = None,
    llm_client: LLMClient | None = None,
    settings: Settings | None = None,
) -> ClarificationRequest | tuple[RequirementSet, CandidatePool]:
    """Run requirements analysis, pause for clarification, then search resumes."""
    settings = settings or get_settings()

    requirements_result = await run_requirements_agent(
        job_id,
        mcp_client,
        recruiter_answers=recruiter_answers,
        llm_client=llm_client,
        settings=settings,
    )

    if isinstance(requirements_result, ClarificationRequest):
        return requirements_result

    candidate_pool = await run_search_agent(
        requirements_result,
        mcp_client,
        settings=settings,
    )

    return requirements_result, candidate_pool
async def run_screening_pipeline(
    job_id: str,
    mcp_client: Client,
    *,
    recruiter_answers: dict[str, Any] | None = None,
    llm_client: LLMClient | None = None,
    settings: Settings | None = None,
) -> (
    ClarificationRequest
    | tuple[
        RequirementSet,
        CandidatePool,
        list[CandidateEvaluation],
        list[ShortlistEntry],
    ]
):
    """Run requirements, search, evaluation, scoring, and ranking."""

    settings = settings or get_settings()

    screening = await prepare_screening(
        job_id,
        mcp_client,
        recruiter_answers=recruiter_answers,
        llm_client=llm_client,
        settings=settings,
    )

    if isinstance(screening, ClarificationRequest):
        return screening

    requirements, candidate_pool = screening

    evaluations = await evaluate_candidates(
        requirements,
        candidate_pool,
        mcp_client,
        llm_client=llm_client,
        settings=settings,
    )

    shortlist = rank_candidates(requirements, evaluations)

    return requirements, candidate_pool, evaluations, shortlist