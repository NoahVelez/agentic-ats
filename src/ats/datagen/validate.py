"""Deterministic checks on seeds, single resumes, and the finished pool."""

from __future__ import annotations

from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from ats.datagen.facts import FactSheet
from ats.datagen.io import JDSeed, RequirementSeed
from ats.datagen.text import (
    EMAIL_RE,
    PHONE_RE,
    URL_RE,
    contains_keyword,
    content_terms,
    content_tokens,
    find_keywords,
    jaccard,
)
from ats.models import CandidateKey, RequirementKey, Resume, Section

DEFAULT_JACCARD_THRESHOLD = 0.2


def text_fields(resume: Resume) -> list[tuple[Section, str, str]]:
    """Every free-text field as ``(section, path, text)``. PII fields are not included."""
    out: list[tuple[Section, str, str]] = [("summary", "summary", resume.summary)]
    for i, e in enumerate(resume.experience):
        out.append(("experience", f"experience[{i}].company", e.company))
        out.append(("experience", f"experience[{i}].title", e.title))
        out.extend(
            ("experience", f"experience[{i}].bullets[{j}]", b) for j, b in enumerate(e.bullets)
        )
    for i, ed in enumerate(resume.education):
        out.append(("education", f"education[{i}].institution", ed.institution))
        out.append(("education", f"education[{i}].degree", ed.degree))
        out.append(("education", f"education[{i}].field", ed.field))
    out.extend(("skills", f"skills[{i}]", s) for i, s in enumerate(resume.skills))
    for i, p in enumerate(resume.projects):
        out.append(("projects", f"projects[{i}].name", p.name))
        out.extend(("projects", f"projects[{i}].bullets[{j}]", b) for j, b in enumerate(p.bullets))
    return out


def resume_text(resume: Resume) -> str:
    return "\n".join(text for _, _, text in text_fields(resume))


def requirement_terms(req: RequirementSeed) -> set[str]:
    return content_tokens(" ".join([req.text, *req.keywords]))


def overlap_score(bullet: str, req: RequirementSeed) -> float:
    """Jaccard between a bullet and the requirement's JD text + keywords (stemmed tokens)."""
    return jaccard(content_tokens(bullet), requirement_terms(req))


# --- per-resume checks ---------------------------------------------------------------------


def check_proofs(resume: Resume, keys: list[RequirementKey]) -> list[str]:
    issues = []
    full = resume_text(resume)
    for key in keys:
        if key.verdict == "missing":
            if key.proof or key.location:
                issues.append(f"{key.requirement_id}: missing verdict must not carry a proof")
            continue
        if not key.proof or not key.location:
            issues.append(f"{key.requirement_id}: {key.verdict} verdict has no proof/location")
            continue
        if key.proof not in full:
            issues.append(f"{key.requirement_id}: proof line not found verbatim in resume")
        loc = key.location
        try:
            at = resume.experience[loc.entry_index].bullets[loc.bullet_index]
        except IndexError:
            at = None
        if loc.section != "experience" or at != key.proof:
            issues.append(f"{key.requirement_id}: proof not at recorded location {loc}")
    return issues


def check_pii(resume: Resume) -> list[str]:
    pii = resume.pii
    street = pii.address.split(",")[0].strip()
    literals = {"name": pii.name, "email": pii.email, "phone": pii.phone, "address": street}
    issues = []
    for _, path, text in text_fields(resume):
        lower = text.lower()
        for label, value in literals.items():
            if value and value.lower() in lower:
                issues.append(f"{path}: contains the candidate's {label}")
        for label, regex in (("email", EMAIL_RE), ("phone number", PHONE_RE), ("URL", URL_RE)):
            if regex.search(text):
                issues.append(f"{path}: looks like it contains a {label}")
    return issues


def check_forbidden_keywords(resume: Resume, facts: FactSheet) -> list[str]:
    """Keywords of requirements the candidate lacks (all keywords for hidden_gem, career_changer,
    keyword_stuffer, filler) may only appear in the skills section. Hidden gems may not list them
    in skills either. Covers the keyword_stuffer and filler rules."""
    issues = []
    for section, path, text in text_fields(resume):
        if section == "skills" and facts.archetype != "hidden_gem":
            continue
        hits = find_keywords(text, facts.forbidden_keywords)
        if hits:
            issues.append(f"{path}: uses forbidden JD keyword(s) {hits}")
    return issues


def check_keyword_stuffer(resume: Resume, jd: JDSeed) -> list[str]:
    skills = "\n".join(resume.skills)
    missing = [kw for kw in jd.all_keywords if not contains_keyword(skills, kw)]
    return [f"skills: keyword_stuffer is missing JD keywords {missing}"] if missing else []


def check_hidden_gem(facts: FactSheet, jd: JDSeed, threshold: float) -> list[str]:
    issues = []
    for proof in facts.proofs:
        score = overlap_score(proof.text, jd.requirement(proof.requirement_id))
        if score >= threshold:
            issues.append(
                f"{proof.requirement_id}: hidden_gem proof overlap {score:.2f} >= {threshold}"
            )
    return issues


def validate_resume(
    resume: Resume,
    keys: list[RequirementKey],
    facts: FactSheet,
    jd: JDSeed,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> list[str]:
    issues = (
        check_proofs(resume, keys) + check_pii(resume) + check_forbidden_keywords(resume, facts)
    )
    if facts.archetype == "keyword_stuffer":
        issues += check_keyword_stuffer(resume, jd)
    if facts.archetype == "hidden_gem":
        issues += check_hidden_gem(facts, jd, jaccard_threshold)
    return issues


# --- seed lint -----------------------------------------------------------------------------


def lint_seed(jd: JDSeed, jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD) -> list[str]:
    """Catch seed mistakes that would make the answer key wrong, before generating anything."""
    issues = []
    ids = [r.id for r in jd.requirements]
    if len(ids) != len(set(ids)):
        issues.append(f"duplicate requirement ids: {ids}")
    for req in jd.requirements:
        others = [kw for r in jd.requirements if r.id != req.id for kw in r.keywords]
        for bullet in req.jd_phrasings:
            if not find_keywords(bullet, req.keywords):
                issues.append(f"{req.id} jd_phrasing has none of its keywords: {bullet!r}")
            if hits := find_keywords(bullet, others):
                issues.append(f"{req.id} jd_phrasing uses other requirements' keywords {hits}")
        for kind in ("paraphrases", "transferable"):
            for bullet in getattr(req, kind):
                if hits := find_keywords(bullet, jd.all_keywords):
                    issues.append(f"{req.id} {kind[:-1]} uses JD keywords {hits}: {bullet!r}")
        for bullet in req.paraphrases:
            score = overlap_score(bullet, req)
            if score >= jaccard_threshold:
                issues.append(
                    f"{req.id} paraphrase overlap {score:.2f} >= {jaccard_threshold}: {bullet!r}"
                )
        for bullet in [*req.jd_phrasings, *req.paraphrases, *req.transferable]:
            for label, regex in (("email", EMAIL_RE), ("phone", PHONE_RE), ("URL", URL_RE)):
                if regex.search(bullet):
                    issues.append(f"{req.id} bullet looks like it contains a {label}: {bullet!r}")
    return issues


# --- pool-level report ---------------------------------------------------------------------


def bm25_ranks(jd_text: str, resumes: Iterable[Resume]) -> dict[str, int]:
    """1-based BM25 rank of every resume for the JD text (ties broken by candidate_id)."""
    resumes = list(resumes)
    if not resumes:
        return {}
    bm25 = BM25Okapi([content_terms(resume_text(r)) for r in resumes])
    scores = bm25.get_scores(content_terms(jd_text))
    order = sorted(
        zip(resumes, scores, strict=True), key=lambda rs: (-float(rs[1]), rs[0].candidate_id)
    )
    return {r.candidate_id: i + 1 for i, (r, _) in enumerate(order)}


def bm25_report(
    jds: list[JDSeed], resumes: list[Resume], keys: list[tuple[str, CandidateKey]]
) -> tuple[dict, list[str]]:
    """Rank each JD's pool with BM25 and report hidden_gem / keyword_stuffer ranks.

    Returns ``(report, warnings)``. A hidden_gem inside the BM25 top-N (N = planted shortlist
    size: clear_fit + hidden_gem + career_changer) is a warning, not a failure.
    """
    report: dict = {}
    warnings: list[str] = []
    for jd in jds:
        pool = [r for r in resumes if jd.job_id in r.job_ids]
        ranks = bm25_ranks(jd.text, pool)
        job_keys = [k for job_id, k in keys if job_id == jd.job_id]
        top_n = sum(1 for k in job_keys if k.expected == "shortlist")
        by_arch: dict[str, dict[str, int]] = {}
        for k in job_keys:
            by_arch.setdefault(k.archetype, {})[k.candidate_id] = ranks[k.candidate_id]
        hidden = dict(sorted(by_arch.get("hidden_gem", {}).items(), key=lambda kv: kv[1]))
        stuffers = dict(sorted(by_arch.get("keyword_stuffer", {}).items(), key=lambda kv: kv[1]))
        job_warnings = [
            f"{jd.job_id}: hidden_gem {cid} is BM25 rank {rank} (inside top-{top_n})"
            for cid, rank in hidden.items()
            if rank <= top_n
        ]
        warnings += job_warnings
        report[jd.job_id] = {
            "pool_size": len(pool),
            "top_n": top_n,
            "hidden_gem_ranks": hidden,
            "keyword_stuffer_ranks": stuffers,
            "warnings": job_warnings,
        }
    return report, warnings
