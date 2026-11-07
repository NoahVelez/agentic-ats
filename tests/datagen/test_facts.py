from __future__ import annotations

import re
from collections import Counter

import pytest

from ats.datagen.facts import DEFAULT_COUNTS, build_fact_sheets, resolve_counts
from ats.datagen.io import JDSeed


@pytest.fixture(scope="module")
def sheets(seed: JDSeed):
    return build_fact_sheets([seed], DEFAULT_COUNTS, seed=42)


def test_deterministic(seed: JDSeed, sheets) -> None:
    again = build_fact_sheets([seed], DEFAULT_COUNTS, seed=42)
    assert [s.model_dump() for s in again] == [s.model_dump() for s in sheets]
    other = build_fact_sheets([seed], DEFAULT_COUNTS, seed=7)
    assert [s.model_dump() for s in other] != [s.model_dump() for s in sheets]


def test_counts_ids_and_shuffle(sheets) -> None:
    assert Counter(s.archetype for s in sheets) == Counter(DEFAULT_COUNTS)
    assert len({s.candidate_id for s in sheets}) == 50
    # IDs must not reveal archetype: the first block is not all one archetype.
    assert len({s.archetype for s in sheets[:6]}) > 1


def test_verdict_rules(seed: JDSeed, sheets) -> None:
    musts = {r.id for r in seed.requirements if r.kind == "must"}
    nices = {r.id for r in seed.requirements if r.kind == "nice"}
    for s in sheets:
        v = s.verdicts
        proofs = {p.requirement_id: p for p in s.proofs}
        assert set(proofs) == {rid for rid, verdict in v.items() if verdict != "missing"}
        for rid, p in proofs.items():
            req = seed.requirement(rid)
            assert p.text in getattr(req, p.source)
            assert 0 <= p.entry_index < len(s.roles)
        if s.archetype == "clear_fit":
            assert all(v[r] == "met" for r in musts)
            assert any(v[r] == "met" for r in nices)
            assert all(p.source == "jd_phrasings" for p in s.proofs)
        elif s.archetype == "hidden_gem":
            assert all(v[r] == "met" for r in musts)
            assert all(p.source == "paraphrases" for p in s.proofs)
        elif s.archetype in ("keyword_stuffer", "filler"):
            assert set(v.values()) == {"missing"}
            assert not s.proofs
        elif s.archetype == "near_miss":
            assert sum(v[r] == "missing" for r in musts) == 1
            assert all(v[r] in ("met", "missing") for r in musts)
            assert {p.source for p in s.proofs} <= {"jd_phrasings", "paraphrases"}
        elif s.archetype == "career_changer":
            assert all(v[r] in ("met", "partial") for r in musts)
            assert any(v[r] == "partial" for r in musts)
            for p in s.proofs:
                assert p.source == ("transferable" if p.verdict == "partial" else "paraphrases")
        if s.archetype == "keyword_stuffer":
            assert sorted(s.stuffed_keywords) == sorted(seed.all_keywords)
        else:
            assert s.stuffed_keywords == []


def test_forbidden_keywords(seed: JDSeed, sheets) -> None:
    for s in sheets:
        if s.archetype in ("hidden_gem", "career_changer", "keyword_stuffer", "filler"):
            assert s.forbidden_keywords == seed.all_keywords
        else:
            missing = [r for r in seed.requirements if s.verdicts[r.id] == "missing"]
            assert set(s.forbidden_keywords) == {k for r in missing for k in r.keywords}


def test_roles_and_pii(sheets) -> None:
    names = [s.pii.name for s in sheets]
    assert len(set(names)) == len(names)
    for s in sheets:
        assert 2 <= len(s.roles) <= 4
        assert s.roles[0].end == "Present"
        assert all(r.start <= (r.end if r.end != "Present" else "9999") for r in s.roles)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.pii.dob)
        assert "@" in s.pii.email


def test_career_changer_bullets_match_field(seed: JDSeed) -> None:
    from ats.datagen.templates import FIELDS
    from ats.datagen.text import find_keywords

    counts = {**{k: 0 for k in DEFAULT_COUNTS}, "career_changer": 30}
    for s in build_fact_sheets([seed], counts, seed=1):
        markers = FIELDS[s.field].markers
        for p in s.proofs:
            options = seed.requirement(p.requirement_id).transferable
            if p.source == "transferable" and any(find_keywords(t, markers) for t in options):
                assert find_keywords(p.text, markers), (s.field, p.text)


def test_resolve_counts_validation() -> None:
    assert resolve_counts({"filler": 3})["filler"] == 3
    with pytest.raises(ValueError):
        resolve_counts({"unicorn": 1})
    with pytest.raises(ValueError):
        resolve_counts({"filler": -1})
