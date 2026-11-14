"""Shared data-contract models (SPEC §10).

Two groups live here: the data-generator contracts (jobs, resumes, answer key) and the contracts
passed between the MCP tools, the agents, and the API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Section = Literal["summary", "experience", "education", "skills", "projects"]
Verdict = Literal["met", "partial", "missing", "unverified"]
Archetype = Literal[
    "clear_fit", "hidden_gem", "keyword_stuffer", "near_miss", "career_changer", "filler"
]
ARCHETYPES: tuple[Archetype, ...] = (
    "clear_fit",
    "hidden_gem",
    "keyword_stuffer",
    "near_miss",
    "career_changer",
    "filler",
)
SHORTLIST_ARCHETYPES: frozenset[Archetype] = frozenset(
    {"clear_fit", "hidden_gem", "career_changer"}
)


class Requirement(BaseModel):
    id: str
    text: str
    kind: Literal["must", "nice"]
    weight: float
    equivalents: list[str] = Field(default_factory=list)
    assumption: str | None = None


class JobDescription(BaseModel):
    job_id: str
    title: str
    text: str


class PII(BaseModel):
    """Personal fields. Kept out of every text field; ingest drops them entirely."""

    name: str
    email: str
    phone: str
    address: str
    dob: str


class ExperienceEntry(BaseModel):
    company: str
    title: str
    start: str = Field(pattern=r"^\d{4}-\d{2}$")
    end: str = Field(pattern=r"^(\d{4}-\d{2}|Present)$")
    bullets: list[str] = Field(min_length=1)


class EducationEntry(BaseModel):
    institution: str
    degree: str
    field: str
    year: int


class ProjectEntry(BaseModel):
    name: str
    bullets: list[str] = Field(min_length=1)


class Resume(BaseModel):
    candidate_id: str
    job_ids: list[str]
    pii: PII
    summary: str
    experience: list[ExperienceEntry]
    education: list[EducationEntry]
    skills: list[str]
    projects: list[ProjectEntry] = Field(default_factory=list)


class ProofLocation(BaseModel):
    section: Section
    entry_index: int
    bullet_index: int


class RequirementKey(BaseModel):
    requirement_id: str
    verdict: Literal["met", "partial", "missing"]
    proof: str | None = None
    location: ProofLocation | None = None


class CandidateKey(BaseModel):
    candidate_id: str
    archetype: Archetype
    expected: Literal["shortlist", "reject"]
    requirements: list[RequirementKey]

    @property
    def met_requirement_ids(self) -> list[str]:
        return [r.requirement_id for r in self.requirements if r.verdict == "met"]


class JobAnswerKey(BaseModel):
    job_id: str
    requirements: list[Requirement]
    shortlist_size: int
    candidates: list[CandidateKey]


class AnswerKey(BaseModel):
    seed: int
    generator: str
    jobs: list[JobAnswerKey]


# --- Requirements -----------------------------------------------------------------------------


class RequirementSet(BaseModel):
    job_id: str
    requirements: list[Requirement]


class Question(BaseModel):
    id: str
    text: str
    about_requirement: str | None = None


class ClarificationRequest(BaseModel):
    questions: list[Question]


class JobSummary(BaseModel):
    job_id: str
    title: str


# --- Search and retrieval ---------------------------------------------------------------------


class Chunk(BaseModel):
    chunk_id: str
    candidate_id: str
    section: Section
    text: str


class SearchHit(BaseModel):
    candidate_id: str
    chunk_id: str
    score: float
    snippet: str


class QueryRound(BaseModel):
    round: int
    query: str
    section: Section | None = None
    candidate_ids: list[str]


class CandidatePool(BaseModel):
    candidate_ids: list[str]
    queries: list[QueryRound]


class ScoreBucket(BaseModel):
    lo: float
    hi: float
    count: int


class PoolStats(BaseModel):
    n_hits: int
    distinct_candidates: int
    score_min: float | None
    score_max: float | None
    # Counts each candidate once, by its best-scoring hit.
    histogram: list[ScoreBucket]
    # None when STRONG_HIT is not configured.
    strong_hit_threshold: float | None
    candidates_with_strong_hit: int | None


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    score: float


# --- Evaluation and verification --------------------------------------------------------------


class Claim(BaseModel):
    requirement_id: str
    verdict: Verdict
    quote: str | None
    chunk_id: str | None
    reasoning: str


class CandidateEvaluation(BaseModel):
    candidate_id: str
    claims: list[Claim]


QuoteReason = Literal["ok", "wrong_candidate", "not_verbatim", "chunk_not_found", "missing_quote"]


class QuoteCheck(BaseModel):
    ok: bool
    reason: QuoteReason


class RejectedClaim(BaseModel):
    claim: Claim
    reason: str


class VerificationResult(BaseModel):
    candidate_id: str
    passed: bool
    rejected: list[RejectedClaim]


# --- Shortlist, review, and trace -------------------------------------------------------------


class Decision(BaseModel):
    candidate_id: str
    action: Literal["approve", "reject", "closer_look"]
    note: str | None = None


class ShortlistEntry(BaseModel):
    rank: int
    candidate_id: str
    score: float
    flags: list[str]
    evaluation: CandidateEvaluation
    decision: Decision | None = None


class TraceEvent(BaseModel):
    ts: datetime
    run_id: str
    stage: str
    agent: str | None
    event: str
    round: int | None
    tool: str | None
    input: dict[str, Any]
    output: dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0
    latency_ms: int
