from __future__ import annotations

import json
from collections import Counter

import pytest

from ats.datagen.facts import DEFAULT_COUNTS
from ats.datagen.io import JDSeed, write_outputs
from ats.datagen.llm import CachedLLMClient
from ats.datagen.pipeline import GenerationConfig, GenerationError, generate
from ats.datagen.render import LLMRenderer, OfflineRenderer
from ats.datagen.validate import check_pii, resume_text
from ats.models import AnswerKey
from scripts.generate_data import main

from .conftest import SEED_PATH, SMALL_COUNTS, FakeLLM, fake_body


def tree_bytes(root) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="module")
def offline_result(seed: JDSeed):
    return generate([seed], OfflineRenderer(42), GenerationConfig(seed=42))


def test_offline_full_pool(seed: JDSeed, offline_result) -> None:
    res = offline_result
    assert len(res.resumes) == 50
    job_key = res.answer_key.jobs[0]
    assert Counter(c.archetype for c in job_key.candidates) == Counter(DEFAULT_COUNTS)
    assert job_key.shortlist_size == 16
    by_id = {r.candidate_id: r for r in res.resumes}
    for cand in job_key.candidates:
        resume = by_id[cand.candidate_id]
        assert check_pii(resume) == []
        for rk in cand.requirements:
            if rk.verdict != "missing":
                assert rk.proof in resume_text(resume)
                loc = rk.location
                assert resume.experience[loc.entry_index].bullets[loc.bullet_index] == rk.proof


def test_offline_bm25_report(offline_result) -> None:
    job = offline_result.report["bm25"]["jd_analytics_mgr"]
    assert job["top_n"] == 16 and job["pool_size"] == 50
    assert len(job["hidden_gem_ranks"]) == 6 and len(job["keyword_stuffer_ranks"]) == 6
    # The offline pool should behave like the adversarial set it is meant to be.
    assert offline_result.warnings == []
    assert max(job["keyword_stuffer_ranks"].values()) <= 16


def test_outputs_are_byte_identical(seed: JDSeed, tmp_path) -> None:
    for name in ("a", "b"):
        res = generate([seed], OfflineRenderer(5), GenerationConfig(seed=5, counts=SMALL_COUNTS))
        write_outputs(tmp_path / name, res.jobs, res.resumes, res.answer_key, res.report)
    a, b = tree_bytes(tmp_path / "a"), tree_bytes(tmp_path / "b")
    assert a == b
    assert {"answer_key.json", "datagen_report.json", "jobs/jd_analytics_mgr.json"} <= set(a)
    AnswerKey.model_validate_json(a["answer_key.json"])


def test_write_outputs_removes_stale_resumes(seed: JDSeed, tmp_path) -> None:
    stale = tmp_path / "resumes" / "cand_9999.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    res = generate([seed], OfflineRenderer(1), GenerationConfig(seed=1, counts=SMALL_COUNTS))
    write_outputs(tmp_path, res.jobs, res.resumes, res.answer_key, res.report)
    assert not stale.exists()
    assert len(list((tmp_path / "resumes").glob("*.json"))) == sum(SMALL_COUNTS.values())


def test_mocked_llm_pipeline_and_cache(seed: JDSeed, tmp_path) -> None:
    config = GenerationConfig(seed=9, counts=SMALL_COUNTS)
    inner = FakeLLM()
    first = generate([seed], LLMRenderer(CachedLLMClient(inner, tmp_path)), config)
    assert first.answer_key.generator == "fake:fake-model"
    assert len(inner.calls) == sum(SMALL_COUNTS.values())

    rerun_inner = FakeLLM()
    second = generate([seed], LLMRenderer(CachedLLMClient(rerun_inner, tmp_path)), config)
    assert rerun_inner.calls == []  # served entirely from the disk cache
    assert [r.model_dump() for r in second.resumes] == [r.model_dump() for r in first.resumes]


def test_regenerates_when_validation_fails(seed: JDSeed) -> None:
    def respond(user, n_roles, call_no):
        leak = None if "previous draft was rejected" in user else "Email me at a.person@mail.com"
        return json.dumps(fake_body(n_roles, extra_bullet=leak))

    client = FakeLLM(respond=respond)
    res = generate([seed], LLMRenderer(client), GenerationConfig(seed=9, counts=SMALL_COUNTS))
    assert set(res.report["regenerated"].values()) == {2}
    assert len(res.report["regenerated"]) == sum(SMALL_COUNTS.values())
    assert all("@" not in resume_text(r) for r in res.resumes)


def test_fails_loudly_after_max_attempts(seed: JDSeed) -> None:
    def respond(user, n_roles, call_no):
        return json.dumps(fake_body(n_roles, extra_bullet="Wrote SQL every day"))

    counts = {**{k: 0 for k in SMALL_COUNTS}, "filler": 1}
    client = FakeLLM(respond=respond)
    with pytest.raises(GenerationError, match="after 3 attempts"):
        generate([seed], LLMRenderer(client), GenerationConfig(seed=9, counts=counts))
    assert len(client.calls) == 3


def test_cli_offline(tmp_path, capsys) -> None:
    args = ["--offline", "--seeds", str(SEED_PATH), "--out", str(tmp_path), "--seed", "3"]
    for name, n in SMALL_COUNTS.items():
        args += [f"--{name.replace('_', '-')}", str(n)]
    assert main(args) == 0
    assert len(list((tmp_path / "resumes").glob("*.json"))) == sum(SMALL_COUNTS.values())
    assert (tmp_path / "answer_key.json").exists()
    assert (tmp_path / "datagen_report.json").exists()
    assert "BM25 ranks" in capsys.readouterr().out


def test_cli_requires_llm_env_unless_offline(tmp_path, monkeypatch, capsys) -> None:
    for var in ("DATAGEN_LLM_PROVIDER", "DATAGEN_LLM_MODEL", "DATAGEN_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert main(["--seeds", str(SEED_PATH), "--out", str(tmp_path)]) == 1
    assert "DATAGEN_LLM_PROVIDER" in capsys.readouterr().err
