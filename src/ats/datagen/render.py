"""Rendering: turn a fact sheet into resume prose, then insert the fixed proof bullets in code.

The renderer (LLM or offline templates) only writes filler. ``assemble_resume`` places the proof
bullets, forces the roles decided in the fact sheet, stuffs keywords for keyword_stuffers, and
records where each proof landed so the answer key can point at it.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from ats.datagen.facts import FactSheet
from ats.datagen.io import JDSeed
from ats.datagen.llm import LLMClient
from ats.datagen.templates import FIELDS, INSTITUTIONS
from ats.datagen.text import find_keywords
from ats.models import (
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    ProofLocation,
    RequirementKey,
    Resume,
)

log = logging.getLogger(__name__)


class ResumeBody(BaseModel):
    """What the renderer produces: resume prose without PII or proof bullets."""

    summary: str = Field(min_length=20)
    experience: list[ExperienceEntry] = Field(min_length=2, max_length=4)
    education: list[EducationEntry] = Field(min_length=1, max_length=3)
    skills: list[str] = Field(min_length=3, max_length=25)
    projects: list[ProjectEntry] = Field(default_factory=list, max_length=3)


class RenderError(RuntimeError):
    pass


class Renderer(Protocol):
    name: str

    def render(
        self, facts: FactSheet, jd: JDSeed, attempt: int, feedback: list[str]
    ) -> ResumeBody: ...


SYSTEM_PROMPT = """\
You write realistic, fictional resume content for a synthetic dataset used to test hiring software.
Rules:
- Output ONLY one JSON object that matches the given JSON schema. No markdown, no commentary.
- Never include personal information: no person names, email addresses, phone numbers, URLs,
  social handles, or street addresses. Company and school names must be fictional.
- Never use any word or phrase from the FORBIDDEN list, in any field, in any letter case.
- Write plain, concrete bullets with believable numbers. Do not mention job descriptions or
  hiring software."""


def build_prompt(facts: FactSheet, jd: JDSeed, feedback: list[str]) -> tuple[str, str]:
    roles = "\n".join(f"  {i}. {r.title} ({r.start} to {r.end})" for i, r in enumerate(facts.roles))
    proof_lines = (
        "\n".join(f"  - role {p.entry_index}: {p.text}" for p in facts.proofs) or "  (none)"
    )
    forbidden = ", ".join(f'"{k}"' for k in facts.forbidden_keywords) or "(none)"
    skills_note = (
        "List 4-8 ordinary skills for this field; more skills are added separately."
        if facts.stuffed_keywords
        else "List 5-10 skills that fit this person's actual background."
    )
    schema = json.dumps(ResumeBody.model_json_schema(), separators=(",", ":"))
    user = f"""\
Write the resume body for one candidate.

Career field: {facts.industry}
Years of experience: {facts.years_experience}
Experience entries (most recent first). Return exactly {len(facts.roles)} entries, in this order,
with these titles and dates; invent a fictional company for each:
{roles}

Write 2-4 bullets per entry about ordinary duties and results in this field. The following
bullets will be inserted into the given roles later by code; do NOT repeat, rephrase, or
contradict them:
{proof_lines}

FORBIDDEN words/phrases (do not use anywhere, including skills): {forbidden}

Summary: 1-2 sentences, consistent with the roles above.
Education: 1-2 entries consistent with the career.
Skills: {skills_note}
Projects: 0-2 short projects relevant to this field.

JSON schema:
{schema}"""
    if feedback:
        issues = "\n".join(f"- {f}" for f in feedback)
        user += f"\n\nA previous draft was rejected for these reasons. Avoid them:\n{issues}"
    return SYSTEM_PROMPT, user


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_body(raw: str, expected_roles: int) -> ResumeBody:
    text = _FENCE_RE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in response")
    body = ResumeBody.model_validate_json(text[start : end + 1])
    if len(body.experience) != expected_roles:
        raise ValueError(
            f"expected exactly {expected_roles} experience entries, got {len(body.experience)}"
        )
    return body


class LLMRenderer:
    def __init__(self, client: LLMClient, max_retries: int = 2) -> None:
        self.client = client
        self.max_retries = max_retries
        provider = getattr(client, "provider", "llm")
        self.name = f"{provider}:{client.model}"

    def render(self, facts: FactSheet, jd: JDSeed, attempt: int, feedback: list[str]) -> ResumeBody:
        system, user = build_prompt(facts, jd, feedback)
        prompt = user
        last_error = ""
        for _ in range(self.max_retries + 1):
            raw = self.client.complete(system, prompt)
            try:
                return parse_body(raw, len(facts.roles))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:800]
                log.info("%s: unparseable LLM output, retrying: %s", facts.candidate_id, last_error)
                prompt = (
                    f"{user}\n\nYour previous reply could not be used: {last_error}\n"
                    "Reply with only the corrected JSON object."
                )
        raise RenderError(
            f"{facts.candidate_id}: LLM output failed validation "
            f"{self.max_retries + 1} times: {last_error}"
        )


class OfflineRenderer:
    """Template-based renderer for CI/tests. Deterministic per (seed, candidate, attempt)."""

    name = "offline"

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def render(self, facts: FactSheet, jd: JDSeed, attempt: int, feedback: list[str]) -> ResumeBody:
        rng = random.Random(f"{self.seed}:{facts.candidate_id}:{attempt}:offline")
        profile = FIELDS[facts.field]
        proof_texts = {p.text.lower() for p in facts.proofs}

        def ok(text: str) -> bool:
            return not find_keywords(text, facts.forbidden_keywords)

        pool = [b for b in profile.bullets if ok(b) and b.lower() not in proof_texts]
        if len(pool) < 2:
            raise RenderError(f"{facts.candidate_id}: too few usable template bullets")
        rng.shuffle(pool)
        companies = list(profile.companies)
        rng.shuffle(companies)

        experience = []
        cursor = 0
        for i, role in enumerate(facts.roles):
            k = rng.randint(2, 3)
            bullets = [pool[(cursor + j) % len(pool)] for j in range(k)]
            cursor += k
            experience.append(
                ExperienceEntry(
                    company=companies[i % len(companies)],
                    title=role.title,
                    start=role.start,
                    end=role.end,
                    bullets=list(dict.fromkeys(bullets)),
                )
            )

        summaries = [s for s in profile.summaries if ok(s)]
        summary = (
            rng.choice(summaries) if summaries else "Professional with {years} years of experience."
        ).format(years=facts.years_experience, industry=facts.industry)

        oldest_start_year = int(facts.roles[-1].start[:4])
        degree, field = rng.choice(profile.degrees)
        education = [
            EducationEntry(
                institution=rng.choice(INSTITUTIONS),
                degree=degree,
                field=field,
                year=oldest_start_year - rng.randint(0, 2),
            )
        ]

        skills = [s for s in profile.skills if ok(s)]
        rng.shuffle(skills)
        skills = skills[: rng.randint(4, 6)]
        if facts.archetype in ("clear_fit", "near_miss"):
            # Qualified candidates written in JD wording may list the matching tool names.
            for req in jd.requirements:
                if facts.verdicts[req.id] == "met":
                    skills.append(req.keywords[0])
        if len(skills) < 3:
            skills += ["teamwork", "time management", "written communication"][: 3 - len(skills)]

        projects = [
            ProjectEntry(name=name, bullets=list(bullets))
            for name, bullets in profile.projects
            if ok(name) and all(ok(b) for b in bullets)
        ]
        rng.shuffle(projects)
        projects = projects[: rng.randint(0, min(2, len(projects)))]

        return ResumeBody(
            summary=summary,
            experience=experience,
            education=education,
            skills=list(dict.fromkeys(skills)),
            projects=projects,
        )


def assemble_resume(
    facts: FactSheet, body: ResumeBody, jd: JDSeed, seed: int, attempt: int
) -> tuple[Resume, list[RequirementKey]]:
    """Insert proof bullets into their assigned roles and build the answer-key rows."""
    rng = random.Random(f"{seed}:{facts.candidate_id}:{attempt}:assemble")
    proof_texts = {p.text.strip().lower() for p in facts.proofs}

    # Each slot is (requirement_id or None, text); None marks renderer-written filler.
    slots: list[list[tuple[str | None, str]]] = []
    for entry in body.experience:
        slots.append([(None, b) for b in entry.bullets if b.strip().lower() not in proof_texts])
    for proof in facts.proofs:
        entry_slots = slots[proof.entry_index]
        entry_slots.insert(rng.randint(0, len(entry_slots)), (proof.requirement_id, proof.text))

    locations: dict[str, ProofLocation] = {}
    experience: list[ExperienceEntry] = []
    for e_idx, (entry, role, entry_slots) in enumerate(
        zip(body.experience, facts.roles, slots, strict=True)
    ):
        for b_idx, (req_id, _) in enumerate(entry_slots):
            if req_id is not None:
                locations[req_id] = ProofLocation(
                    section="experience", entry_index=e_idx, bullet_index=b_idx
                )
        experience.append(
            ExperienceEntry(
                company=entry.company,
                title=role.title,
                start=role.start,
                end=role.end,
                bullets=[text for _, text in entry_slots],
            )
        )

    skills = list(body.skills)
    if facts.stuffed_keywords:
        have = {s.lower() for s in skills}
        skills += [kw for kw in facts.stuffed_keywords if kw.lower() not in have]
        rng.shuffle(skills)

    resume = Resume(
        candidate_id=facts.candidate_id,
        job_ids=[facts.job_id],
        pii=facts.pii,
        summary=body.summary,
        experience=experience,
        education=body.education,
        skills=skills,
        projects=body.projects,
    )
    keys = []
    for req in jd.requirements:
        proof = facts.proof_for(req.id)
        keys.append(
            RequirementKey(
                requirement_id=req.id,
                verdict=facts.verdicts[req.id],
                proof=proof.text if proof else None,
                location=locations.get(req.id),
            )
        )
    return resume, keys
