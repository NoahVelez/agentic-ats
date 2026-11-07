from __future__ import annotations

import hashlib
import json

import pytest

from ats.datagen import llm as llm_mod
from ats.datagen.facts import build_fact_sheets
from ats.datagen.io import JDSeed
from ats.datagen.llm import CachedLLMClient, ConfigError, client_from_env
from ats.datagen.render import (
    LLMRenderer,
    RenderError,
    ResumeBody,
    assemble_resume,
    build_prompt,
    parse_body,
)

from .conftest import FakeLLM, fake_body


@pytest.fixture(scope="module")
def sheets(seed: JDSeed):
    counts = {
        "clear_fit": 2,
        "hidden_gem": 1,
        "keyword_stuffer": 1,
        "near_miss": 1,
        "career_changer": 1,
        "filler": 1,
    }
    return build_fact_sheets([seed], counts, seed=11)


def by_arch(sheets, archetype):
    return next(s for s in sheets if s.archetype == archetype)


def test_assemble_inserts_proofs_at_recorded_location(seed: JDSeed, sheets) -> None:
    for facts in sheets:
        body = ResumeBody.model_validate(fake_body(len(facts.roles)))
        resume, keys = assemble_resume(facts, body, seed, seed=11, attempt=1)
        assert [(e.title, e.start, e.end) for e in resume.experience] == [
            (r.title, r.start, r.end) for r in facts.roles
        ]
        for key in keys:
            proof = facts.proof_for(key.requirement_id)
            if proof is None:
                assert key.verdict == "missing" and key.location is None
                continue
            loc = key.location
            assert loc.entry_index == proof.entry_index
            assert resume.experience[loc.entry_index].bullets[loc.bullet_index] == proof.text
        filler_count = sum(len(e.bullets) for e in body.experience)
        assert sum(len(e.bullets) for e in resume.experience) == filler_count + len(facts.proofs)


def test_assemble_stuffs_keywords(seed: JDSeed, sheets) -> None:
    facts = by_arch(sheets, "keyword_stuffer")
    body = ResumeBody.model_validate(fake_body(len(facts.roles)))
    resume, _ = assemble_resume(facts, body, seed, seed=11, attempt=1)
    assert set(seed.all_keywords) <= set(resume.skills)


def test_prompt_lists_forbidden_and_has_no_pii(seed: JDSeed, sheets) -> None:
    facts = by_arch(sheets, "hidden_gem")
    system, user = build_prompt(facts, seed, [])
    assert '"SQL"' in user and '"Tableau"' in user
    for value in (facts.pii.name, facts.pii.email, facts.pii.phone):
        assert value not in system + user
    assert "previous draft" not in user
    _, retry = build_prompt(facts, seed, ["summary: uses forbidden JD keyword(s) ['SQL']"])
    assert "previous draft was rejected" in retry and "['SQL']" in retry


def test_parse_body_handles_fences_and_role_count() -> None:
    raw = "```json\n" + json.dumps(fake_body(3)) + "\n```"
    assert len(parse_body(raw, 3).experience) == 3
    with pytest.raises(ValueError, match="exactly 2"):
        parse_body(raw, 2)


def test_llm_renderer_retries_then_succeeds(seed: JDSeed, sheets) -> None:
    facts = by_arch(sheets, "filler")

    def respond(user, n_roles, call_no):
        return "not json" if call_no == 1 else json.dumps(fake_body(n_roles))

    client = FakeLLM(respond=respond)
    body = LLMRenderer(client).render(facts, seed, 1, [])
    assert len(body.experience) == len(facts.roles)
    assert len(client.calls) == 2
    assert "could not be used" in client.calls[1][1]


def test_llm_renderer_gives_up_after_two_retries(seed: JDSeed, sheets) -> None:
    facts = by_arch(sheets, "filler")
    client = FakeLLM(respond=lambda *_: '{"summary": "too short"}')
    with pytest.raises(RenderError):
        LLMRenderer(client).render(facts, seed, 1, [])
    assert len(client.calls) == 3


def test_disk_cache(tmp_path) -> None:
    inner = FakeLLM(respond=lambda user, n, c: json.dumps({"call": c}))
    cached = CachedLLMClient(inner, tmp_path)
    user = "Return exactly 2 entries"
    first = cached.complete("sys", user)
    assert cached.complete("sys", user) == first
    assert len(inner.calls) == 1
    expected = hashlib.sha256(("fake-model" + "sys\n\n" + user).encode()).hexdigest()
    assert (tmp_path / f"{expected}.json").exists()

    other = CachedLLMClient(FakeLLM(model="other", respond=lambda *a: "x"), tmp_path)
    assert other.complete("sys", user) == "x"  # different model -> different key


def test_client_from_env_validation(monkeypatch, tmp_path) -> None:
    with pytest.raises(ConfigError, match="DATAGEN_LLM_MODEL"):
        client_from_env({"DATAGEN_LLM_PROVIDER": "openai", "DATAGEN_LLM_API_KEY": "k"})
    with pytest.raises(ConfigError, match="unknown"):
        client_from_env(
            {"DATAGEN_LLM_PROVIDER": "nope", "DATAGEN_LLM_MODEL": "m", "DATAGEN_LLM_API_KEY": "k"}
        )

    class Dummy:
        def __init__(self, model, api_key):
            self.model = model

        def complete(self, system, user):
            return "{}"

    monkeypatch.setitem(llm_mod.PROVIDERS, "dummy", Dummy)
    env = {"DATAGEN_LLM_PROVIDER": "dummy", "DATAGEN_LLM_MODEL": "m1", "DATAGEN_LLM_API_KEY": "k"}
    client = client_from_env(env, cache_dir=tmp_path)
    assert isinstance(client, CachedLLMClient)
    assert LLMRenderer(client).name == "dummy:m1"
    assert isinstance(client_from_env(env, cache_dir=None), Dummy)
