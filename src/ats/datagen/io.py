"""Seed loading and deterministic output writing."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ats.models import AnswerKey, JobDescription, Requirement, Resume


class RequirementSeed(BaseModel):
    id: str
    text: str
    kind: str = Field(pattern="^(must|nice)$")
    weight: float = Field(gt=0)
    jd_phrasings: list[str] = Field(min_length=1)
    paraphrases: list[str] = Field(min_length=1)
    transferable: list[str] = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)

    def to_requirement(self) -> Requirement:
        return Requirement(
            id=self.id,
            text=self.text,
            kind=self.kind,  # type: ignore[arg-type]
            weight=self.weight,
            equivalents=list(self.keywords),
        )


class JDSeed(BaseModel):
    job_id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    title: str
    text: str
    # Career field (key in templates.FIELDS) used for clear_fit / hidden_gem / near_miss.
    field: str = "analytics"
    requirements: list[RequirementSeed] = Field(min_length=1)

    @property
    def all_keywords(self) -> list[str]:
        seen: dict[str, None] = {}
        for req in self.requirements:
            for kw in req.keywords:
                seen.setdefault(kw, None)
        return list(seen)

    def requirement(self, req_id: str) -> RequirementSeed:
        for req in self.requirements:
            if req.id == req_id:
                return req
        raise KeyError(req_id)

    def to_job(self) -> JobDescription:
        return JobDescription(job_id=self.job_id, title=self.title, text=self.text)


def load_seed(path: Path) -> JDSeed:
    with Path(path).open(encoding="utf-8") as fh:
        return JDSeed.model_validate(yaml.safe_load(fh))


def load_seeds(paths: list[Path]) -> list[JDSeed]:
    """Load seeds from files and/or directories (``*.yaml``/``*.yml``), sorted by path."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted([*p.glob("*.yaml"), *p.glob("*.yml")]))
        else:
            files.append(p)
    seeds = [load_seed(f) for f in files]
    ids = [s.job_id for s in seeds]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate job_id across seeds: {ids}")
    if not seeds:
        raise ValueError(f"no JD seeds found in {[str(p) for p in paths]}")
    return seeds


def dumps(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _clear_json(directory: Path) -> None:
    if directory.is_dir():
        for f in directory.glob("*.json"):
            f.unlink()


def write_outputs(
    out_dir: Path,
    jobs: list[JobDescription],
    resumes: list[Resume],
    answer_key: AnswerKey,
    report: dict,
) -> None:
    """Write ``jobs/*.json``, ``resumes/*.json``, ``answer_key.json`` and ``datagen_report.json``.

    Previously generated job/resume JSON files are removed first so a smaller rerun leaves no
    stale candidates behind.
    """
    out_dir = Path(out_dir)
    _clear_json(out_dir / "jobs")
    _clear_json(out_dir / "resumes")
    for job in jobs:
        write_text(out_dir / "jobs" / f"{job.job_id}.json", dumps(job))
    for resume in resumes:
        write_text(out_dir / "resumes" / f"{resume.candidate_id}.json", dumps(resume))
    write_text(out_dir / "answer_key.json", dumps(answer_key))
    write_text(
        out_dir / "datagen_report.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
