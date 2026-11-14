"""Generator-specific text helpers: token overlap and keyword matching.

Tokenizing and the PII patterns are shared with ingest and live in ``ats.text``.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ats.text import EMAIL_RE, PHONE_RE, STOPWORDS, URL_RE, content_terms, stem, tokenize

__all__ = [
    "EMAIL_RE",
    "PHONE_RE",
    "STOPWORDS",
    "URL_RE",
    "contains_keyword",
    "content_terms",
    "content_tokens",
    "find_keywords",
    "jaccard",
    "stem",
    "tokenize",
]


def content_tokens(text: str) -> set[str]:
    return set(content_terms(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


@lru_cache(maxsize=4096)
def _keyword_re(keyword: str) -> re.Pattern[str]:
    # Lookarounds instead of \b so keywords that start/end with punctuation (C++, A/B) still work.
    return re.compile(r"(?<![\w])" + re.escape(keyword) + r"(?![\w])", re.IGNORECASE)


def contains_keyword(text: str, keyword: str) -> bool:
    return _keyword_re(keyword).search(text) is not None


def find_keywords(text: str, keywords: list[str] | tuple[str, ...]) -> list[str]:
    return [kw for kw in keywords if contains_keyword(text, kw)]
