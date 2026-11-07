from __future__ import annotations

from ats.datagen.io import JDSeed
from ats.datagen.validate import lint_seed


def test_seed_shape(seed: JDSeed) -> None:
    assert 6 <= len(seed.requirements) <= 8
    assert sum(r.kind == "must" for r in seed.requirements) >= 4
    for req in seed.requirements:
        assert len(req.jd_phrasings) >= 3, req.id
        assert len(req.paraphrases) >= 3, req.id
        assert len(req.transferable) >= 2, req.id
        assert req.keywords, req.id


def test_seed_lints_clean(seed: JDSeed) -> None:
    assert lint_seed(seed) == []


def test_lint_catches_keyword_in_paraphrase(seed: JDSeed) -> None:
    bad = seed.model_copy(deep=True)
    bad.requirements[0].paraphrases.append("Wrote SQL reports for the sales group")
    issues = lint_seed(bad)
    assert any("paraphrase uses JD keywords" in i for i in issues)


def test_lint_catches_jd_phrasing_without_keyword(seed: JDSeed) -> None:
    bad = seed.model_copy(deep=True)
    bad.requirements[0].jd_phrasings.append("Helped out around the office")
    assert any("has none of its keywords" in i for i in lint_seed(bad))


def test_lint_catches_high_overlap_paraphrase(seed: JDSeed) -> None:
    bad = seed.model_copy(deep=True)
    # Shares most tokens with "People management experience" but avoids exact keyword phrases.
    bad.requirements[0].paraphrases.append("People managing experience")
    assert any("paraphrase overlap" in i for i in lint_seed(bad))
