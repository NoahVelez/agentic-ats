"""Text helpers shared by the data generator, ingest, and the baselines.

The PII patterns live here so the generator's validator and ingest's redaction use the same
definitions: text the generator accepts is text that redaction leaves untouched.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or over our per so that
    the their them they this to was were which while with within without i me my we us you your
    he she his her after before across each every all any more most than then via up down out
    about also using used use who whom what when where why how not no can could would should
    will may might must""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/+#][a-z0-9]+)*")
_SUFFIXES = ("ments", "ment", "ings", "ing", "ers", "er", "ies", "es", "ed", "s")

EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\b[\w\-]+\.(?:com|org|io|edu|gov|dev|co|ai|me)(?:/\S*)?\b",
    re.IGNORECASE,
)
PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", EMAIL_RE),
    ("phone", PHONE_RE),
    ("url", URL_RE),
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def content_terms(text: str) -> list[str]:
    """Stemmed, stopword-free tokens (keeps duplicates; used for BM25)."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]
