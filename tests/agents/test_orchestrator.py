
from unittest.mock import AsyncMock

import pytest

from ats.models import (
    CandidateEvaluation,
    CandidatePool,
    Claim,
    ClarificationRequest,
    Question,
    Requirement,
    RequirementSet,
)
from ats.orchestrator import (
    rank_candidates,
    run_screening_pipeline,
    score_candidate,
)


def make_requirements():
    return RequirementSet(
        job_id="job-001",
        requirements=[
            Requirement(
                id="r1",
                text="Python experience",
                kind="must",
                weight=3,
            ),
            Requirement(
                id="r2",
                text="SQL experience",
                kind="nice",
                weight=2,
            ),
        ],
    )


def test_score_candidate():
    requirements = make_requirements()
    evaluation = CandidateEvaluation(
        candidate_id="candidate-001",
        claims=[
            Claim(
                requirement_id="r1",
                verdict="met",
                quote="Used Python for data analysis",
                chunk_id="chunk-1",
                reasoning="Direct evidence",
            ),
            Claim(
                requirement_id="r2",
                verdict="partial",
                quote="Worked with relational databases",
                chunk_id="chunk-2",
                reasoning="Partial evidence",
            ),
        ],
    )

    score, flags = score_candidate(requirements, evaluation)
    assert score == 4.0
    assert flags == []


def test_missing_must_have_adds_flag():
    requirements = make_requirements()
    evaluation = CandidateEvaluation(
        candidate_id="candidate-002",
        claims=[
            Claim(
                requirement_id="r1",
                verdict="missing",
                quote=None,
                chunk_id=None,
                reasoning="No supporting evidence",
            ),
            Claim(
                requirement_id="r2",
                verdict="met",
                quote="Used SQL",
                chunk_id="chunk-3",
                reasoning="Direct evidence",
            ),
        ],
    )

    score, flags = score_candidate(requirements, evaluation)
    assert score == 2.0
    assert flags == ["missing_must_have"]


def test_unverified_claim_does_not_contribute():
    requirements = make_requirements()
    evaluation = CandidateEvaluation(
        candidate_id="candidate-003",
        claims=[
            Claim(
                requirement_id="r1",
                verdict="unverified",
                quote=None,
                chunk_id=None,
                reasoning="Verification failed",
            ),
        ],
    )

    score, flags = score_candidate(requirements, evaluation)
    assert score == 0.0
    assert flags == []


def test_rank_candidates_places_missing_must_have_last():
    requirements = make_requirements()

    eligible = CandidateEvaluation(
        candidate_id="candidate-001",
        claims=[
            Claim(
                requirement_id="r1",
                verdict="met",
                quote="Python experience",
                chunk_id="chunk-1",
                reasoning="Evidence found",
            ),
        ],
    )

    missing_must = CandidateEvaluation(
        candidate_id="candidate-002",
        claims=[
            Claim(
                requirement_id="r1",
                verdict="missing",
                quote=None,
                chunk_id=None,
                reasoning="No evidence",
            ),
            Claim(
                requirement_id="r2",
                verdict="met",
                quote="SQL experience",
                chunk_id="chunk-2",
                reasoning="Evidence found",
            ),
        ],
    )

    shortlist = rank_candidates(requirements, [missing_must, eligible])

    assert shortlist[0].candidate_id == "candidate-001"
    assert shortlist[1].candidate_id == "candidate-002"
    assert "missing_must_have" in shortlist[1].flags


@pytest.mark.asyncio
async def test_pipeline_returns_clarification_request(monkeypatch):
    clarification = ClarificationRequest(
        questions=[
            Question(
                id="q1",
                text="How many years of Python experience are required?",
            )
        ]
    )

    mock_prepare = AsyncMock(return_value=clarification)

    monkeypatch.setattr(
        "ats.orchestrator.prepare_screening",
        mock_prepare,
    )

    result = await run_screening_pipeline(
        "job-001",
        AsyncMock(),
    )

    assert isinstance(result, ClarificationRequest)
    assert len(result.questions) == 1
    mock_prepare.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_returns_ranked_shortlist(monkeypatch):
    requirements = make_requirements()

    pool = CandidatePool(
        candidate_ids=["candidate-001"],
        queries=[],
    )

    evaluation = CandidateEvaluation(
        candidate_id="candidate-001",
        claims=[
            Claim(
                requirement_id="r1",
                verdict="met",
                quote="Used Python",
                chunk_id="chunk-1",
                reasoning="Direct evidence",
            ),
        ],
    )

    async def mock_prepare(*args, **kwargs):
        return requirements, pool

    async def mock_evaluate(*args, **kwargs):
        return [evaluation]

    monkeypatch.setattr(
        "ats.orchestrator.prepare_screening",
        mock_prepare,
    )
    monkeypatch.setattr(
        "ats.orchestrator.evaluate_candidates",
        mock_evaluate,
    )

    result = await run_screening_pipeline(
        "job-001",
        AsyncMock(),
    )

    assert result[0] == requirements
    assert result[1] == pool
    assert len(result[2]) == 1
    assert result[3][0].candidate_id == "candidate-001"
    assert result[3][0].score == 3.0
