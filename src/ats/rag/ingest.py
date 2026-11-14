"""Ingest: chunk, redact, embed, and index the generated resumes (SPEC §6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ats.data import DataError, load_answer_key, load_jobs, load_resumes
from ats.models import AnswerKey, Chunk, Resume
from ats.rag.chunking import MAX_CHUNK_TOKENS, bullet_line, chunk_resume, estimate_tokens
from ats.rag.redaction import redact_text
from ats.rag.store import ResumeStore


@dataclass
class IngestReport:
    embedder: str
    n_jobs: int
    n_resumes: int
    n_chunks: int
    largest_chunk_tokens: int
    # candidate_id -> kinds of PII that redaction had to remove. Empty for clean generator output.
    redactions: dict[str, list[str]] = field(default_factory=dict)
    # None when there is no answer key to check against.
    missing_proofs: list[str] | None = None


def chunk_and_redact(
    resume: Resume, max_tokens: int = MAX_CHUNK_TOKENS
) -> tuple[list[Chunk], list[str]]:
    """Chunks with PII removed, plus the kinds of PII that were found in the text."""
    chunks: list[Chunk] = []
    kinds: list[str] = []
    for chunk in chunk_resume(resume, max_tokens):
        text, found = redact_text(chunk.text, resume.pii)
        chunks.append(chunk.model_copy(update={"text": text}))
        kinds += [k for k in found if k not in kinds]
    return chunks, kinds


def find_missing_proofs(answer_key: AnswerKey, chunks: list[Chunk]) -> list[str]:
    """Answer-key proof lines that are not a whole, unchanged bullet in one of that candidate's
    chunks, in the section the key records. Any entry here means the dataset and the index
    disagree."""
    by_candidate: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_candidate.setdefault(chunk.candidate_id, []).append(chunk)
    problems: list[str] = []
    for job in answer_key.jobs:
        for cand in job.candidates:
            own = by_candidate.get(cand.candidate_id)
            if own is None:
                problems.append(f"{cand.candidate_id}: in the answer key but not in the index")
                continue
            for req in cand.requirements:
                if not req.proof:
                    continue
                section = req.location.section if req.location else "experience"
                wanted = bullet_line(req.proof)
                if not any(c.section == section and wanted in c.text.splitlines() for c in own):
                    problems.append(
                        f"{cand.candidate_id} {req.requirement_id}: proof line not found verbatim"
                    )
    return problems


def ingest(data_dir: Path, store: ResumeStore, max_tokens: int = MAX_CHUNK_TOKENS) -> IngestReport:
    data_dir = Path(data_dir)
    jobs = load_jobs(data_dir)
    resumes = load_resumes(data_dir)
    if not jobs or not resumes:
        raise DataError(f"no jobs or resumes under {data_dir}; run scripts/generate_data.py first")
    ids = [r.candidate_id for r in resumes]
    if len(ids) != len(set(ids)):
        raise DataError("duplicate candidate_id in resumes")
    for resume in resumes:
        unknown = sorted(set(resume.job_ids) - set(jobs))
        if unknown:
            raise DataError(f"{resume.candidate_id} refers to unknown job(s) {unknown}")

    chunks: list[Chunk] = []
    redactions: dict[str, list[str]] = {}
    for resume in resumes:
        own, kinds = chunk_and_redact(resume, max_tokens)
        chunks += own
        if kinds:
            redactions[resume.candidate_id] = kinds

    store.rebuild(chunks, {r.candidate_id: r.job_ids for r in resumes})

    answer_key = load_answer_key(data_dir)
    return IngestReport(
        embedder=store.embedder.name,
        n_jobs=len(jobs),
        n_resumes=len(resumes),
        n_chunks=len(chunks),
        largest_chunk_tokens=max(estimate_tokens(c.text) for c in chunks),
        redactions=redactions,
        missing_proofs=find_missing_proofs(answer_key, chunks) if answer_key else None,
    )
