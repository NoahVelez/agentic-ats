from __future__ import annotations

import pytest

from ats.datagen.facts import build_fact_sheets
from ats.datagen.io import JDSeed
from ats.datagen.render import OfflineRenderer, assemble_resume
from ats.datagen.text import contains_keyword, find_keywords
from ats.datagen.validate import (
    bm25_ranks,
    bm25_report,
    check_forbidden_keywords,
    check_hidden_gem,
    check_keyword_stuffer,
    check_pii,
    check_proofs,
    overlap_score,
    validate_resume,
)
from ats.models import CandidateKey

ONE_EACH = {
    "clear_fit": 1,
    "hidden_gem": 1,
    "keyword_stuffer": 1,
    "near_miss": 1,
    "career_changer": 1,
    "filler": 1,
}


@pytest.fixture(scope="module")
def built(seed: JDSeed):
    out = {}
    for facts in build_fact_sheets([seed], ONE_EACH, seed=3):
        body = OfflineRenderer(3).render(facts, seed, 1, [])
        resume, keys = assemble_resume(facts, body, seed, 3, 1)
        out[facts.archetype] = (facts, resume, keys)
    return out


@pytest.mark.parametrize(
    ("text", "keyword", "expected"),
    [
        ("Wrote PostgreSQL joins", "SQL", False),
        ("Wrote T-SQL procedures", "SQL", True),
        ("ran sql daily", "SQL", True),
        ("Ran A/B tests weekly", "A/B test", False),
        ("Ran A/B tests weekly", "A/B tests", True),
        ("Built dashboards", "dashboard", False),
    ],
)
def test_keyword_matching(text: str, keyword: str, expected: bool) -> None:
    assert contains_keyword(text, keyword) is expected


def test_offline_resumes_pass_all_checks(seed: JDSeed, built) -> None:
    for archetype, (facts, resume, keys) in built.items():
        assert validate_resume(resume, keys, facts, seed) == [], archetype


def test_pii_detection(built) -> None:
    _, resume, _ = built["filler"]
    pii = resume.pii
    for leak in (
        f"Worked with {pii.name} on audits",
        f"Reach me at {pii.email}",
        f"Call {pii.phone}",
        "Contact: someone.else@example.org",
        "Call 312-555-0199 for references",
        "Portfolio at https://portfolio.example",
        "See www.mysite.net",
        "Blog: janedoe.dev",
    ):
        bad = resume.model_copy(deep=True)
        bad.experience[0].bullets.append(leak)
        assert check_pii(bad), leak
    assert check_pii(resume) == []


def test_pii_regex_ignores_dates_and_numbers(built) -> None:
    _, resume, _ = built["filler"]
    ok = resume.model_copy(deep=True)
    ok.experience[0].bullets.append("From 2019-2021 handled 1,200 orders and $4M of stock (99.5%)")
    assert check_pii(ok) == []


def test_proof_checks(built) -> None:
    _, resume, keys = built["clear_fit"]
    assert check_proofs(resume, keys) == []
    tampered = resume.model_copy(deep=True)
    loc = next(k.location for k in keys if k.location)
    tampered.experience[loc.entry_index].bullets[loc.bullet_index] += " (edited)"
    assert check_proofs(tampered, keys)


def test_missing_keywords_only_in_skills(seed: JDSeed, built) -> None:
    facts, resume, _ = built["filler"]
    in_skills = resume.model_copy(deep=True)
    in_skills.skills.append("SQL")
    assert check_forbidden_keywords(in_skills, facts) == []
    in_text = resume.model_copy(deep=True)
    in_text.summary += " Comfortable with SQL."
    assert check_forbidden_keywords(in_text, facts)


def test_keyword_stuffer(seed: JDSeed, built) -> None:
    facts, resume, _ = built["keyword_stuffer"]
    assert check_keyword_stuffer(resume, seed) == []
    assert all(find_keywords(s, seed.all_keywords) == [] for s in [resume.summary])
    stripped = resume.model_copy(deep=True)
    stripped.skills = [s for s in stripped.skills if s != "Tableau"]
    assert check_keyword_stuffer(stripped, seed)
    leaked = resume.model_copy(deep=True)
    leaked.experience[0].bullets.append("Built Tableau views for the shop")
    assert check_forbidden_keywords(leaked, facts)


def test_hidden_gem_rules(seed: JDSeed, built) -> None:
    facts, resume, _ = built["hidden_gem"]
    assert check_hidden_gem(facts, seed, 0.2) == []
    assert check_hidden_gem(facts, seed, 0.0)  # any threshold of 0 flags every bullet
    skills_leak = resume.model_copy(deep=True)
    skills_leak.skills.append("Power BI")
    assert check_forbidden_keywords(skills_leak, facts)


def test_jd_phrasings_overlap_more_than_paraphrases(seed: JDSeed) -> None:
    for req in seed.requirements:
        jd = sum(overlap_score(b, req) for b in req.jd_phrasings) / len(req.jd_phrasings)
        para = sum(overlap_score(b, req) for b in req.paraphrases) / len(req.paraphrases)
        assert jd > para, req.id


def test_bm25_ranks_and_warning(seed: JDSeed, built) -> None:
    resumes = [r for _, r, _ in built.values()]
    ranks = bm25_ranks(seed.text, resumes)
    assert sorted(ranks.values()) == list(range(1, len(resumes) + 1))
    assert ranks[built["keyword_stuffer"][1].candidate_id] == 1

    # A "hidden gem" that copies the JD should trigger the (non-fatal) warning.
    facts, gem, keys = built["hidden_gem"]
    loud = gem.model_copy(deep=True)
    loud.summary = seed.text * 3
    pool = [loud if r.candidate_id == gem.candidate_id else r for r in resumes]
    key_rows = [
        (
            seed.job_id,
            CandidateKey(
                candidate_id=f.candidate_id,
                archetype=f.archetype,
                expected=f.expected,
                requirements=k,
            ),
        )
        for f, _, k in built.values()
    ]
    report, warnings = bm25_report([seed], pool, key_rows)
    job = report[seed.job_id]
    assert job["top_n"] == 3  # clear_fit + hidden_gem + career_changer
    assert job["hidden_gem_ranks"] == {gem.candidate_id: 1}
    assert warnings and gem.candidate_id in warnings[0]
