"""Deterministic quote checking behind the ``verify_quote`` tool (SPEC §4.4, step 1).

A quote passes only if it is a verbatim substring of the named chunk after whitespace and
quote-mark normalization, and the chunk belongs to the candidate being evaluated. Matching stays
case-sensitive: a quote that changes the wording is not verbatim.
"""

from __future__ import annotations

from ats.models import Chunk, QuoteCheck

_QUOTE_MAP = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u0060": "'",
        "\u00b4": "'",
    }
)


def normalize(text: str) -> str:
    """Straighten curly quotes and collapse every run of whitespace to one space."""
    return " ".join(text.translate(_QUOTE_MAP).split())


def check_quote(chunk: Chunk | None, candidate_id: str, quote: str | None) -> QuoteCheck:
    if chunk is None:
        return QuoteCheck(ok=False, reason="chunk_not_found")
    if chunk.candidate_id != candidate_id:
        # Decided before the text is looked at, so nothing about another candidate leaks.
        return QuoteCheck(ok=False, reason="wrong_candidate")
    wanted = normalize(quote or "")
    if not wanted:
        return QuoteCheck(ok=False, reason="missing_quote")
    if wanted not in normalize(chunk.text):
        return QuoteCheck(ok=False, reason="not_verbatim")
    return QuoteCheck(ok=True, reason="ok")
