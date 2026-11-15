from __future__ import annotations

import math
from pathlib import Path

import pytest

from ats.data import load_answer_key
from ats.datagen.facts import DEFAULT_COUNTS
from ats.rag.store import ResumeStore
from scripts.calibrate import auc, calibrate, main, percentile, render


def test_percentile_and_auc() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 25) == 2.0
    assert percentile([4.0, 1.0], 50) == 2.5
    assert percentile([7.0], 25) == 7.0
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5, 0.5], [0.5]) == 0.5
    assert math.isnan(auc([], [0.5]))


def test_calibration_on_the_offline_dataset(store: ResumeStore, dataset: Path) -> None:
    key = load_answer_key(dataset)
    result = calibrate(store, key)
    n_proofs = sum(bool(r.proof) for c in key.jobs[0].candidates for r in c.requirements)
    assert len(result.proofs) == n_proofs > 100
    assert result.embedder == "hash:bow-512" and result.section is None
    assert 0.0 <= result.threshold < 1.0
    # The 25th percentile: at least about three quarters of the proof chunks are at or above it
    # (more when scores tie, which a word-matching embedder does a lot at zero).
    share = sum(score >= result.threshold for *_, score in result.proofs) / n_proofs
    assert share >= 0.70
    # Word matching favours JD wording: clear fits' proof lines outscore hidden gems' paraphrases.
    by_archetype: dict[str, list[float]] = {}
    for archetype, _, _, score in result.proofs:
        by_archetype.setdefault(archetype, []).append(score)
    assert percentile(by_archetype["clear_fit"], 50) > percentile(by_archetype["hidden_gem"], 50)

    assert [r.requirement_id for r in result.reach] == [f"R{i}" for i in range(1, 8)]
    for row in result.reach:
        assert row.has_it + row.lacks_it == 50
        assert row.has_it_reached <= row.has_it and row.lacks_it_reached <= row.lacks_it
        assert 0.0 <= row.auc <= 1.0
    assert {a: total for a, (_, total) in result.any_strong_hit.items()} == DEFAULT_COUNTS

    text = render(result)
    assert f"STRONG_HIT = {result.threshold}" in text and "all sections" in text
    assert "hidden_gem" in text and "verdict: partial" in text

    # A higher percentile gives a bar at least as high, and searching one section is recorded.
    assert calibrate(store, key, q=75).threshold > result.threshold
    experience = calibrate(store, key, section="experience")
    assert experience.section == "experience" and len(experience.proofs) == n_proofs
    assert "experience chunks only" in render(experience)


def test_calibrate_cli(indexed, dataset: Path, tmp_path, monkeypatch, capsys) -> None:
    for var in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env
    args = ["--offline", "--data-dir", str(dataset), "--chroma-dir", str(indexed[0])]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Embedder: hash:bow-512" in out and "STRONG_HIT = 0." in out

    assert main([*args, "--section", "experience"]) == 0
    assert "experience chunks only" in capsys.readouterr().out

    empty = tmp_path / "no_data"
    empty.mkdir()
    assert main(["--offline", "--data-dir", str(empty), "--chroma-dir", str(indexed[0])]) == 1
    assert "answer_key.json" in capsys.readouterr().err
    assert main(["--offline", "--data-dir", str(dataset), "--chroma-dir", str(tmp_path / "x")]) == 1
    assert "scripts/ingest.py" in capsys.readouterr().err
    # Without --offline and with no provider configured, it explains what to set.
    assert main(["--data-dir", str(dataset), "--chroma-dir", str(indexed[0])]) == 1
    assert "EMBEDDING_PROVIDER is not set" in capsys.readouterr().err


@pytest.mark.parametrize("q", [10.0, 50.0, 90.0])
def test_other_percentiles(store: ResumeStore, dataset: Path, q: float) -> None:
    result = calibrate(store, load_answer_key(dataset), q=q)
    share = sum(score >= result.threshold for *_, score in result.proofs) / len(result.proofs)
    # At least the expected share clears the bar; ties can only push the share up.
    assert share >= (1 - q / 100) - 0.06
