"""PII redaction for chunk text at ingest, and the response filter for tool output (SPEC §5.3).

Two layers:
- ``redact_text`` replaces a candidate's own PII values (from the resume's ``pii`` field) and
  anything shaped like an email, phone number, or URL.
- ``scrub`` is the second check on what tools return: it drops PII-named fields and redacts the
  same patterns in every string.

Neither layer can recognise the name of some *other* person written inside a bullet. Only a
candidate's own recorded name is known here.
"""

from __future__ import annotations

import re
from typing import Any

from ats.models import PII
from ats.text import PII_PATTERNS

REDACTED = "[REDACTED]"
PII_FIELDS = frozenset({"name", "email", "phone", "address", "photo_url", "dob"})


def _literal_patterns(pii: PII) -> list[tuple[str, re.Pattern[str]]]:
    street = pii.address.split(",")[0].strip()
    values = [
        ("name", pii.name),
        ("email", pii.email),
        ("phone", pii.phone),
        ("address", pii.address),
        ("address", street),
        ("dob", pii.dob),
    ]
    patterns = []
    # Longest first, so a full address is replaced before its street line.
    for kind, value in sorted(values, key=lambda kv: -len(kv[1])):
        value = value.strip()
        if not value:
            continue
        # Tolerate different spacing between the words of a name or address.
        body = r"\s+".join(re.escape(part) for part in value.split())
        patterns.append((kind, re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)))
    return patterns


def redact_text(text: str, pii: PII | None = None) -> tuple[str, list[str]]:
    """Return ``(redacted_text, kinds_found)``. ``kinds_found`` is empty when nothing changed."""
    found: list[str] = []
    patterns = (_literal_patterns(pii) if pii else []) + list(PII_PATTERNS)
    for kind, pattern in patterns:
        text, n = pattern.subn(REDACTED, text)
        if n and kind not in found:
            found.append(kind)
    return text, found


def scrub(value: Any, _path: str = "") -> tuple[Any, list[str]]:
    """Recursively drop PII-named keys and redact PII-shaped text. Returns ``(clean, findings)``."""
    findings: list[str] = []
    if isinstance(value, str):
        clean, kinds = redact_text(value)
        findings += [f"{_path or '.'}: {kind}" for kind in kinds]
        return clean, findings
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in PII_FIELDS:
                findings.append(f"{_path}.{key}: field removed")
                continue
            out[key], found = scrub(item, f"{_path}.{key}")
            findings += found
        return out, findings
    if isinstance(value, list | tuple):
        items = []
        for i, item in enumerate(value):
            clean, found = scrub(item, f"{_path}[{i}]")
            items.append(clean)
            findings += found
        return items, findings
    return value, findings
