from __future__ import annotations

import pytest

from ats.models import ExperienceEntry
from ats.rag.ingest import chunk_and_redact
from ats.rag.quotes import check_quote, normalize
from ats.rag.redaction import REDACTED, redact_text, scrub

from .test_chunking import make_resume

PII = make_resume().pii


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Worked with Jordan Avery on audits", "name"),
        ("Worked with JORDAN   AVERY on audits", "name"),
        ("Reach me at jordan.avery@example.org", "email"),
        ("Call (312) 555-0142 any time", "phone"),
        ("Based at 12 Elm Street, Springfield, IL 62701", "address"),
        ("Based at 12 Elm Street near the park", "address"),
        ("Born 1990-04-02", "dob"),
    ],
)
def test_redacts_the_candidates_own_pii(text: str, kind: str) -> None:
    clean, found = redact_text(text, PII)
    assert REDACTED in clean and kind in found
    assert "Avery" not in clean and "555" not in clean and "Elm" not in clean


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Contact someone.else@mail.com for references", "email"),
        ("References: 312-555-0199", "phone"),
        ("Portfolio at https://portfolio.example/work", "url"),
        ("Blog: janedoe.dev", "url"),
    ],
)
def test_redacts_pii_shaped_text_without_knowing_the_candidate(text: str, kind: str) -> None:
    clean, found = redact_text(text)
    assert REDACTED in clean and found == [kind]


def test_clean_text_is_returned_unchanged() -> None:
    text = "From 2019-2021 handled 1,200 orders and $4M of stock (99.5%) using A/B tests"
    assert redact_text(text, PII) == (text, [])


def test_other_peoples_names_are_not_detected() -> None:
    # Known limit: only the candidate's own recorded name can be recognised.
    assert redact_text("Reported to Priya Raman", PII) == ("Reported to Priya Raman", [])


def test_name_planted_in_a_bullet_is_redacted_at_ingest() -> None:
    resume = make_resume(
        experience=[
            ExperienceEntry(
                company="Acme",
                title="Analyst",
                start="2020-01",
                end="Present",
                bullets=["Jordan Avery led the audit", "Built weekly reports"],
            )
        ]
    )
    chunks, kinds = chunk_and_redact(resume)
    text = "\n".join(c.text for c in chunks)
    assert kinds == ["name"]
    assert "Jordan Avery" not in text and f"- {REDACTED} led the audit" in text
    assert "- Built weekly reports" in text


def test_scrub_drops_pii_fields_and_redacts_nested_strings() -> None:
    value = {
        "candidate_id": "cand_1",
        "name": "Jordan Avery",
        "nested": [{"text": "mail a.b@c.org", "dob": "1990-04-02", "score": 0.5}],
        "ok": True,
    }
    clean, findings = scrub(value)
    assert clean == {
        "candidate_id": "cand_1",
        "nested": [{"text": f"mail {REDACTED}", "score": 0.5}],
        "ok": True,
    }
    assert findings == [
        ".name: field removed",
        ".nested[0].text: email",
        ".nested[0].dob: field removed",
    ]
    assert all("Avery" not in f and "c.org" not in f for f in findings)


def test_scrub_leaves_clean_values_alone() -> None:
    value = {"chunk_id": "cand_1:skills:0", "text": "Skills: Excel, SQL", "n": [1, 2.5, None]}
    assert scrub(value) == (value, [])


def test_normalize_handles_whitespace_and_quote_marks() -> None:
    assert normalize("  Led  a\n\tteam ") == "Led a team"
    assert normalize("\u201cdata\u201d team\u2019s") == '"data" team\'s'


def test_check_quote_reasons() -> None:
    [chunk] = [c for c in chunk_and_redact(make_resume())[0] if c.section == "experience"]

    def reason(candidate_id: str, quote: str | None, chunk=chunk) -> str:
        result = check_quote(chunk, candidate_id, quote)
        assert result.ok is (result.reason == "ok")
        return result.reason

    assert reason("cand_test", "Built weekly revenue reports") == "ok"
    assert reason("cand_test", "Built  weekly\nrevenue reports") == "ok"
    assert (
        reason("cand_test", "- Built weekly revenue reports - Mentored two junior analysts") == "ok"
    )
    assert reason("cand_test", "built weekly revenue reports") == "not_verbatim"
    assert reason("cand_test", "Built weekly reports") == "not_verbatim"
    assert reason("cand_test", "   ") == "missing_quote"
    assert reason("cand_test", None) == "missing_quote"
    assert reason("cand_other", "Built weekly revenue reports") == "wrong_candidate"
    assert reason("cand_test", "anything", chunk=None) == "chunk_not_found"
