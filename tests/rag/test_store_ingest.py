from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest

from ats.data import DataError, load_answer_key
from ats.models import Chunk
from ats.rag.baselines import bm25_scores, embedding_baseline, keyword_baseline
from ats.rag.chunking import MAX_CHUNK_TOKENS, chunk_resume
from ats.rag.embeddings import EmbeddingError, HashEmbedder
from ats.rag.ingest import find_missing_proofs, ingest
from ats.rag.store import EmbedderMismatch, ResumeStore, StoreNotReady
from scripts.ingest import main as ingest_main

from ..conftest import JOB_ID


def chunk(candidate_id: str, n: int, text: str, section: str = "experience") -> Chunk:
    return Chunk(
        chunk_id=f"{candidate_id}:{section}:{n}",
        candidate_id=candidate_id,
        section=section,
        text=text,
    )


def test_ingest_report(indexed, resumes) -> None:
    _, report = indexed
    assert report.n_resumes == len(resumes) == 50 and report.n_jobs == 1
    assert report.n_chunks == sum(len(chunk_resume(r)) for r in resumes)
    assert report.largest_chunk_tokens <= MAX_CHUNK_TOKENS
    assert report.embedder == "hash:bow-512"
    # Generator output is already PII-free, so redaction has nothing to do and proofs survive.
    assert report.redactions == {}
    assert report.missing_proofs == []


def test_every_proof_line_is_verbatim_in_one_chunk(store: ResumeStore, answer_key) -> None:
    checked = 0
    for cand in answer_key.candidates:
        chunks = store.candidate_chunks(cand.candidate_id)
        for req in cand.requirements:
            if req.proof:
                holders = [c for c in chunks if f"- {req.proof}" in c.text.splitlines()]
                assert len(holders) == 1 and holders[0].section == "experience"
                checked += 1
    assert checked > 100


def test_candidate_chunks_keep_resume_order(store: ResumeStore, resumes) -> None:
    for resume in resumes[:10]:
        assert store.candidate_chunks(resume.candidate_id) == chunk_resume(resume)
    only = store.candidate_chunks(resumes[0].candidate_id, ["skills", "summary"])
    assert [c.section for c in only] == ["summary", "skills"]
    assert store.candidate_chunks("cand_9999") == []


def test_search_scores_filters_and_scoping(store: ResumeStore) -> None:
    hits = store.search("people management of a team of analysts", 25)
    assert len(hits) == 25
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True) and all(0.0 <= s <= 1.0 for s in scores)
    assert len({c.chunk_id for c, _ in hits}) == 25

    assert {c.section for c, _ in store.search("Excel", 30, section="skills")} == {"skills"}
    target = hits[0][0].candidate_id
    own = store.search("dashboards", 50, candidate_id=target)
    assert own and {c.candidate_id for c, _ in own} == {target}
    assert len(own) == len(store.candidate_chunks(target))


def test_lookups(store: ResumeStore, resumes) -> None:
    first = chunk_resume(resumes[0])[0]
    assert store.get_chunk(first.chunk_id) == first
    assert store.get_chunk("cand_0001:experience:999") is None
    assert store.has_candidate(resumes[0].candidate_id) and not store.has_candidate("cand_9999")
    assert store.counts() == {"chunks": 339, "candidates": 50}
    docs = store.documents(JOB_ID)
    assert list(docs) == sorted(r.candidate_id for r in resumes)
    assert (
        store.documents("no_such_job") == {} and store.rank_by_embedding("x", "no_such_job") == []
    )


def test_baselines_rank_the_whole_pool(store: ResumeStore, dataset: Path, answer_key) -> None:
    job_text = (dataset / "jobs" / f"{JOB_ID}.json").read_text(encoding="utf-8")
    archetype = {c.candidate_id: c.archetype for c in answer_key.candidates}
    n = answer_key.shortlist_size
    for baseline in (keyword_baseline, embedding_baseline):
        everyone = baseline(store, JOB_ID, job_text, 1000)
        assert [r.rank for r in everyone] == list(range(1, 51))
        assert len({r.candidate_id for r in everyone}) == 50
        scores = [r.score for r in everyone]
        assert scores == sorted(scores, reverse=True)
        assert baseline(store, JOB_ID, job_text, 5) == everyone[:5]
    # The keyword baseline falls for the trap the dataset sets: every stuffer in, no hidden gem.
    top = Counter(archetype[r.candidate_id] for r in keyword_baseline(store, JOB_ID, job_text, n))
    assert top["keyword_stuffer"] == 6 and top["hidden_gem"] == 0


def test_bm25_scores_order_and_ties() -> None:
    docs = {
        "b": "wrote sql for dashboards",
        "a": "wrote sql for dashboards",
        "c": "framed houses and roofs",
        "d": "cooked dinner service nightly",
        "e": "designed tea packaging",
        "f": "taught algebra to students",
    }
    ranked = bm25_scores("sql dashboards", docs)
    assert [cid for cid, _ in ranked[:2]] == ["a", "b"] and ranked[0][1] == ranked[1][1] > 0
    assert [(cid, score) for cid, score in ranked[2:]] == [(c, 0.0) for c in "cdef"]
    assert bm25_scores("anything", {}) == []


def test_store_not_ready_and_embedder_mismatch(indexed, tmp_path) -> None:
    for path in (tmp_path / "missing", tmp_path):
        empty = ResumeStore(path, HashEmbedder())
        assert not empty.is_ready()
        with pytest.raises(StoreNotReady, match="scripts/ingest.py"):
            empty.search("x", 1)
    assert not (tmp_path / "missing").exists()

    other = ResumeStore(indexed[0], HashEmbedder(dim=256))
    assert not other.is_ready()
    with pytest.raises(EmbedderMismatch, match="hash:bow-512.*hash:bow-256"):
        other.search("x", 1)


def test_rebuild_replaces_everything(tmp_path) -> None:
    store = ResumeStore(tmp_path, HashEmbedder())
    store.rebuild(
        [chunk("c1", 0, "Built SQL reports"), chunk("c2", 0, "Framed houses")],
        {"c1": ["j1"], "c2": ["j1", "j2"]},
    )
    assert store.counts() == {"chunks": 2, "candidates": 2}
    assert list(store.documents("j2")) == ["c2"]
    store.rebuild([chunk("c3", 0, "Cooked dinner service")], {"c3": ["j1"]})
    assert store.counts() == {"chunks": 1, "candidates": 1}
    assert store.get_chunk("c1:experience:0") is None and list(store.documents("j1")) == ["c3"]


def test_failed_embedding_keeps_the_previous_index(tmp_path) -> None:
    class Flaky(HashEmbedder):
        fail = False

        def embed(self, texts):
            if self.fail:
                raise EmbeddingError("provider down")
            return super().embed(texts)

    embedder = Flaky()
    store = ResumeStore(tmp_path, embedder)
    store.rebuild([chunk("c1", 0, "Built SQL reports")], {"c1": ["j1"]})
    embedder.fail = True
    with pytest.raises(EmbeddingError):
        store.rebuild([chunk("c2", 0, "Framed houses")], {"c2": ["j1"]})
    embedder.fail = False
    assert store.get_chunk("c1:experience:0") is not None and store.counts()["chunks"] == 1


def test_rebuild_input_validation(tmp_path) -> None:
    store = ResumeStore(tmp_path, HashEmbedder())
    with pytest.raises(ValueError, match="no chunks"):
        store.rebuild([], {})
    with pytest.raises(ValueError, match="no job_ids"):
        store.rebuild([chunk("c1", 0, "text")], {})


def test_find_missing_proofs_detects_a_changed_line(dataset: Path, resumes) -> None:
    key = load_answer_key(dataset)
    chunks = [c for r in resumes for c in chunk_resume(r)]
    assert find_missing_proofs(key, chunks) == []

    victim = next(c for c in key.jobs[0].candidates if any(r.proof for r in c.requirements))
    proof = next(r for r in victim.requirements if r.proof)
    tampered = [
        c.model_copy(update={"text": c.text.replace(proof.proof, proof.proof + " (edited)")})
        if c.candidate_id == victim.candidate_id
        else c
        for c in chunks
    ]
    problems = find_missing_proofs(key, tampered)
    assert problems == [
        f"{victim.candidate_id} {proof.requirement_id}: proof line not found verbatim"
    ]
    without = [c for c in chunks if c.candidate_id != victim.candidate_id]
    assert f"{victim.candidate_id}: in the answer key but not in the index" in find_missing_proofs(
        key, without
    )


def test_ingest_rejects_bad_datasets(dataset: Path, tmp_path) -> None:
    store = ResumeStore(tmp_path / "chroma", HashEmbedder())
    with pytest.raises(DataError, match="generate_data.py"):
        ingest(tmp_path / "empty", store)
    broken = tmp_path / "broken"
    shutil.copytree(dataset, broken)
    (broken / "jobs" / f"{JOB_ID}.json").unlink()
    (broken / "jobs" / "other.json").write_text(
        '{"job_id": "other", "title": "Other", "text": "x"}', encoding="utf-8"
    )
    with pytest.raises(DataError, match="unknown job"):
        ingest(broken, store)


def test_ingest_cli(dataset: Path, tmp_path, monkeypatch, capsys) -> None:
    for var in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env
    chroma = tmp_path / "chroma"
    args = ["--data-dir", str(dataset), "--chroma-dir", str(chroma)]

    assert ingest_main([*args, "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Indexed 50 resumes (339 chunks" in out and "word for word" in out
    assert ResumeStore(chroma, HashEmbedder()).counts()["candidates"] == 50

    assert ingest_main(args) == 1
    assert "EMBEDDING_PROVIDER is not set" in capsys.readouterr().err
    assert (
        ingest_main(
            ["--offline", "--data-dir", str(tmp_path / "nope"), "--chroma-dir", str(chroma)]
        )
        == 1
    )
    assert "generate_data.py" in capsys.readouterr().err
