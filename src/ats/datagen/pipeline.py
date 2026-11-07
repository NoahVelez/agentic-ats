"""End-to-end generation: facts -> render -> assemble -> validate (regenerate) -> report."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ats.datagen.facts import FactSheet, build_fact_sheets, resolve_counts
from ats.datagen.io import JDSeed
from ats.datagen.render import Renderer, RenderError, assemble_resume
from ats.datagen.validate import (
    DEFAULT_JACCARD_THRESHOLD,
    bm25_report,
    lint_seed,
    validate_resume,
)
from ats.models import AnswerKey, CandidateKey, JobAnswerKey, JobDescription, Resume

log = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    pass


@dataclass
class GenerationConfig:
    seed: int = 42
    counts: dict[str, int] = field(default_factory=lambda: resolve_counts())
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD
    max_attempts: int = 3
    workers: int = 4


@dataclass
class GenerationResult:
    jobs: list[JobDescription]
    resumes: list[Resume]
    answer_key: AnswerKey
    report: dict
    warnings: list[str]


def generate_candidate(
    facts: FactSheet, jd: JDSeed, renderer: Renderer, config: GenerationConfig
) -> tuple[Resume, CandidateKey, int]:
    """Render and validate one candidate, regenerating up to ``max_attempts`` times."""
    feedback: list[str] = []
    for attempt in range(1, config.max_attempts + 1):
        try:
            body = renderer.render(facts, jd, attempt, feedback)
        except RenderError as exc:
            feedback = [str(exc)]
            log.warning("%s attempt %d: render failed: %s", facts.candidate_id, attempt, exc)
            continue
        resume, req_keys = assemble_resume(facts, body, jd, config.seed, attempt)
        issues = validate_resume(resume, req_keys, facts, jd, config.jaccard_threshold)
        if not issues:
            key = CandidateKey(
                candidate_id=facts.candidate_id,
                archetype=facts.archetype,
                expected=facts.expected,
                requirements=req_keys,
            )
            return resume, key, attempt
        feedback = issues
        log.warning(
            "%s (%s) attempt %d rejected: %s",
            facts.candidate_id,
            facts.archetype,
            attempt,
            "; ".join(issues),
        )
    raise GenerationError(
        f"{facts.candidate_id} ({facts.archetype}) failed validation after "
        f"{config.max_attempts} attempts: {'; '.join(feedback)}"
    )


def generate(
    seeds: list[JDSeed], renderer: Renderer, config: GenerationConfig | None = None
) -> GenerationResult:
    config = config or GenerationConfig()
    lint_issues = [
        f"{jd.job_id}: {issue}" for jd in seeds for issue in lint_seed(jd, config.jaccard_threshold)
    ]
    if lint_issues:
        raise GenerationError("JD seed problems:\n  " + "\n  ".join(lint_issues))

    sheets = build_fact_sheets(seeds, config.counts, config.seed)
    by_job = {jd.job_id: jd for jd in seeds}

    def run(facts: FactSheet) -> tuple[Resume, CandidateKey, int]:
        return generate_candidate(facts, by_job[facts.job_id], renderer, config)

    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        results = list(pool.map(run, sheets))

    resumes = [r for r, _, _ in results]
    keyed = [(f.job_id, k) for f, (_, k, _) in zip(sheets, results, strict=True)]
    regenerated = {r.candidate_id: n for r, _, n in results if n > 1}

    jobs_key = []
    for jd in seeds:
        cands = [k for job_id, k in keyed if job_id == jd.job_id]
        jobs_key.append(
            JobAnswerKey(
                job_id=jd.job_id,
                requirements=[r.to_requirement() for r in jd.requirements],
                shortlist_size=sum(1 for k in cands if k.expected == "shortlist"),
                candidates=sorted(cands, key=lambda k: k.candidate_id),
            )
        )
    answer_key = AnswerKey(seed=config.seed, generator=renderer.name, jobs=jobs_key)

    bm25, warnings = bm25_report(seeds, resumes, keyed)
    report = {
        "seed": config.seed,
        "generator": renderer.name,
        "counts": config.counts,
        "regenerated": regenerated,
        "bm25": bm25,
    }
    return GenerationResult(
        jobs=[jd.to_job() for jd in seeds],
        resumes=resumes,
        answer_key=answer_key,
        report=report,
        warnings=warnings,
    )
