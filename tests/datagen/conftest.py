from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ats.datagen.io import JDSeed, load_seed

ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = ROOT / "data" / "jd_seeds" / "analytics_manager.yaml"

SMALL_COUNTS = {
    "clear_fit": 2,
    "hidden_gem": 2,
    "keyword_stuffer": 2,
    "near_miss": 2,
    "career_changer": 2,
    "filler": 3,
}


@pytest.fixture(scope="session")
def seed() -> JDSeed:
    return load_seed(SEED_PATH)


def fake_body(n_roles: int, extra_bullet: str | None = None) -> dict:
    experience = []
    for i in range(n_roles):
        bullets = [
            f"Handled day-to-day operations for site {i + 1}",
            "Kept weekly records tidy and up to date",
        ]
        if extra_bullet and i == 0:
            bullets.append(extra_bullet)
        experience.append(
            {
                "company": f"Acme Holdings {i + 1}",
                "title": "Placeholder",
                "start": "2020-01",
                "end": "2021-01",
                "bullets": bullets,
            }
        )
    return {
        "summary": "Dependable professional with a steady track record of delivery.",
        "experience": experience,
        "education": [
            {"institution": "Pinecrest College", "degree": "B.A.", "field": "History", "year": 2012}
        ],
        "skills": ["Excel", "scheduling", "organization"],
        "projects": [],
    }


class FakeLLM:
    """Answers with a neutral, valid ResumeBody sized to the roles requested in the prompt."""

    def __init__(self, model: str = "fake-model", respond=None) -> None:
        self.model = model
        self.provider = "fake"
        self.calls: list[tuple[str, str]] = []
        self._respond = respond

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        n_roles = int(re.search(r"Return exactly (\d+) entries", user).group(1))
        if self._respond is not None:
            return self._respond(user, n_roles, len(self.calls))
        return json.dumps(fake_body(n_roles))


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
