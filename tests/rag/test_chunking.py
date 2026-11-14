from __future__ import annotations

import re
from collections import Counter

from ats.models import PII, ExperienceEntry, ProjectEntry, Resume
from ats.rag.chunking import MAX_CHUNK_TOKENS, chunk_resume, estimate_tokens

SECTION_ORDER = ["summary", "experience", "education", "skills", "projects"]


def make_resume(**overrides) -> Resume:
    base = dict(
        candidate_id="cand_test",
        job_ids=["job_x"],
        pii=PII(
            name="Jordan Avery",
            email="jordan.avery@example.org",
            phone="(312) 555-0142",
            address="12 Elm Street, Springfield, IL 62701",
            dob="1990-04-02",
        ),
        summary="Analyst with 6 years of experience.",
        experience=[
            ExperienceEntry(
                company="Acme Data",
                title="Senior Analyst",
                start="2021-01",
                end="Present",
                bullets=["Built weekly revenue reports", "Mentored two junior analysts"],
            )
        ],
        education=[],
        skills=["Excel", "forecasting"],
        projects=[],
    )
    return Resume(**{**base, **overrides})


def test_ids_sections_and_numbering(resumes) -> None:
    for resume in resumes:
        chunks = chunk_resume(resume)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        seen_sections = list(dict.fromkeys(c.section for c in chunks))
        assert seen_sections == [s for s in SECTION_ORDER if s in seen_sections]
        for section in seen_sections:
            numbers = [
                int(re.fullmatch(rf"{resume.candidate_id}:{section}:(\d+)", c.chunk_id).group(1))
                for c in chunks
                if c.section == section
            ]
            assert numbers == list(range(len(numbers)))
        assert all(c.candidate_id == resume.candidate_id for c in chunks)


def test_every_bullet_is_whole_and_none_is_lost_or_added(resumes) -> None:
    # Counted, not just looked up: template resumes can repeat a filler bullet across roles.
    for resume in resumes:
        lines = Counter(
            line
            for c in chunk_resume(resume)
            if c.section == "experience"
            for line in c.text.splitlines()
            if line.startswith("- ")
        )
        expected = Counter(f"- {b}" for entry in resume.experience for b in entry.bullets)
        assert lines == expected, resume.candidate_id


def test_content_is_kept_and_budget_holds(resumes) -> None:
    for resume in resumes:
        chunks = chunk_resume(resume)
        text = "\n".join(c.text for c in chunks)
        assert resume.summary in text
        assert all(skill in text for skill in resume.skills)
        assert all(e.company in text and e.title in text for e in resume.experience)
        assert all(p.name in text for p in resume.projects)
        assert max(estimate_tokens(c.text) for c in chunks) <= MAX_CHUNK_TOKENS


def test_pii_never_enters_chunk_text(resumes) -> None:
    for resume in resumes:
        text = "\n".join(c.text for c in chunk_resume(resume)).lower()
        pii = resume.pii
        street = pii.address.split(",")[0]
        for value in (pii.name, pii.email, pii.phone, street, pii.dob):
            assert value.lower() not in text, (resume.candidate_id, value)


def test_long_entry_splits_at_bullets_and_repeats_header() -> None:
    bullets = [
        f"Delivered project number {i} on time and under budget for the team" for i in range(20)
    ]
    resume = make_resume(
        experience=[
            ExperienceEntry(
                company="Acme Data", title="Lead", start="2020-01", end="Present", bullets=bullets
            )
        ]
    )
    chunks = [c for c in chunk_resume(resume, max_tokens=60) if c.section == "experience"]
    assert len(chunks) > 1
    assert all(c.text.startswith("Lead, Acme Data (2020-01 to Present)\n") for c in chunks)
    assert all(estimate_tokens(c.text) <= 60 for c in chunks)
    assert [line[2:] for c in chunks for line in c.text.splitlines()[1:]] == bullets


def test_oversized_bullet_stays_whole() -> None:
    long_bullet = "Coordinated " + ", ".join(f"workstream {i}" for i in range(60))
    resume = make_resume(
        experience=[
            ExperienceEntry(
                company="Acme", title="Lead", start="2020-01", end="Present", bullets=[long_bullet]
            )
        ]
    )
    [chunk] = [c for c in chunk_resume(resume, max_tokens=40) if c.section == "experience"]
    assert long_bullet in chunk.text


def test_skills_and_projects_split() -> None:
    resume = make_resume(
        skills=[f"skill number {i}" for i in range(80)],
        projects=[
            ProjectEntry(name="Churn model", bullets=["Flagged at-risk accounts"]),
            ProjectEntry(name="Glossary", bullets=["Documented 60 metrics"]),
        ],
    )
    chunks = chunk_resume(resume)
    skills = [c for c in chunks if c.section == "skills"]
    assert len(skills) > 1 and all(c.text.startswith("Skills: ") for c in skills)
    listed = [s for c in skills for s in c.text.removeprefix("Skills: ").split(", ")]
    assert listed == resume.skills
    assert [c.text for c in chunks if c.section == "projects"] == [
        "Churn model\n- Flagged at-risk accounts",
        "Glossary\n- Documented 60 metrics",
    ]


def test_empty_sections_produce_no_chunks() -> None:
    chunks = chunk_resume(make_resume(education=[], projects=[], skills=[]))
    assert {c.section for c in chunks} == {"summary", "experience"}
