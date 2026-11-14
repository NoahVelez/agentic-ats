"""Loaders for the generated dataset (``data/generated`` by default)."""

from __future__ import annotations

from pathlib import Path

from ats.models import AnswerKey, JobDescription, Resume


class DataError(RuntimeError):
    pass


def load_jobs(data_dir: Path) -> dict[str, JobDescription]:
    files = sorted((Path(data_dir) / "jobs").glob("*.json"))
    jobs = [JobDescription.model_validate_json(f.read_text(encoding="utf-8")) for f in files]
    return {job.job_id: job for job in jobs}


def load_resumes(data_dir: Path) -> list[Resume]:
    files = sorted((Path(data_dir) / "resumes").glob("*.json"))
    return [Resume.model_validate_json(f.read_text(encoding="utf-8")) for f in files]


def load_answer_key(data_dir: Path) -> AnswerKey | None:
    """The answer key, or None if absent. Only ingest's self-check and the eval may call this;
    nothing reachable from an MCP tool does."""
    path = Path(data_dir) / "answer_key.json"
    if not path.is_file():
        return None
    return AnswerKey.model_validate_json(path.read_text(encoding="utf-8"))
