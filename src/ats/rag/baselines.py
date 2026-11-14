"""Whole-resume baselines (SPEC §11): keyword (BM25) and embedding (cosine) ranking."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from ats.models import RankedCandidate
from ats.rag.store import ResumeStore
from ats.text import content_terms


def bm25_scores(query: str, documents: dict[str, str]) -> list[tuple[str, float]]:
    """Score every document against the query; highest first, ties broken by id."""
    if not documents:
        return []
    ids = list(documents)
    bm25 = BM25Okapi([content_terms(documents[i]) for i in ids])
    scores = bm25.get_scores(content_terms(query))
    return sorted(
        ((i, float(s)) for i, s in zip(ids, scores, strict=True)),
        key=lambda item: (-item[1], item[0]),
    )


def _ranked(scored: list[tuple[str, float]], top_n: int) -> list[RankedCandidate]:
    return [
        RankedCandidate(rank=i + 1, candidate_id=cid, score=round(score, 6))
        for i, (cid, score) in enumerate(scored[:top_n])
    ]


def keyword_baseline(
    store: ResumeStore, job_id: str, job_text: str, top_n: int
) -> list[RankedCandidate]:
    return _ranked(bm25_scores(job_text, store.documents(job_id)), top_n)


def embedding_baseline(
    store: ResumeStore, job_id: str, job_text: str, top_n: int
) -> list[RankedCandidate]:
    return _ranked(store.rank_by_embedding(job_text, job_id), top_n)
