"""Shared data-contract models (SPEC §10).

Only the models produced by the synthetic data generator live here for now: job descriptions,
requirements, resumes, and the answer key. The agent/run models are added later.
"""

from __future__ import annotations

from typing import Literal

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
