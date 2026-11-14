"""Section-aware chunking (SPEC §6.2).

One chunk per summary, experience entry, education block, skills list, and project. Bullets are
never split. A block that exceeds the token budget is split at bullet boundaries, repeating its
header so each piece still says which role it belongs to.

Chunk text is built only from resume content fields. ``Resume.pii`` is never read here.
"""

from __future__ import annotations

import re

from ats.models import Chunk, Resume, Section

MAX_CHUNK_TOKENS = 120

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Rough token count (words plus punctuation marks); close enough for a ~120 budget."""
    return len(_TOKEN_RE.findall(text))


def _split_block(header: str | None, lines: list[str], max_tokens: int) -> list[str]:
    """Pack ``lines`` under ``header`` into as few texts as fit the budget. Lines stay whole."""
    head = [header] if header else []
    texts: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and estimate_tokens("\n".join([*head, *current, line])) > max_tokens:
            texts.append("\n".join([*head, *current]))
            current = []
        current.append(line)
    if current:
        texts.append("\n".join([*head, *current]))
    return texts


def _split_items(prefix: str, items: list[str], max_tokens: int) -> list[str]:
    texts: list[str] = []
    current: list[str] = []
    for item in items:
        if current and estimate_tokens(prefix + ", ".join([*current, item])) > max_tokens:
            texts.append(prefix + ", ".join(current))
            current = []
        current.append(item)
    if current:
        texts.append(prefix + ", ".join(current))
    return texts


def experience_header(title: str, company: str, start: str, end: str) -> str:
    return f"{title}, {company} ({start} to {end})"


def bullet_line(bullet: str) -> str:
    return f"- {bullet}"


def chunk_resume(resume: Resume, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    """Split one resume into chunks with ids ``{candidate_id}:{section}:{n}``."""
    sections: list[tuple[Section, list[str]]] = []

    if resume.summary.strip():
        sections.append(("summary", [resume.summary.strip()]))

    experience: list[str] = []
    for entry in resume.experience:
        header = experience_header(entry.title, entry.company, entry.start, entry.end)
        experience += _split_block(header, [bullet_line(b) for b in entry.bullets], max_tokens)
    sections.append(("experience", experience))

    education = [f"{e.degree} in {e.field}, {e.institution} ({e.year})" for e in resume.education]
    sections.append(("education", _split_block(None, education, max_tokens)))

    sections.append(("skills", _split_items("Skills: ", resume.skills, max_tokens)))

    projects: list[str] = []
    for project in resume.projects:
        projects += _split_block(
            project.name, [bullet_line(b) for b in project.bullets], max_tokens
        )
    sections.append(("projects", projects))

    return [
        Chunk(
            chunk_id=f"{resume.candidate_id}:{section}:{n}",
            candidate_id=resume.candidate_id,
            section=section,
            text=text,
        )
        for section, texts in sections
        for n, text in enumerate(texts)
    ]
