"""Deterministic fact sheets: the single source of truth for the answer key.

Everything here is driven by one ``random.Random(seed)`` plus a seeded Faker instance, so the
same seed always yields the same archetypes, verdicts, proof bullets, roles, and PII.
"""

from __future__ import annotations

import random
import re
from datetime import date
from typing import Literal

from faker import Faker
from pydantic import BaseModel

from ats.datagen.io import JDSeed, RequirementSeed
from ats.datagen.templates import CHANGER_FIELDS, FIELDS, FILLER_FIELDS, STUFFER_FIELDS
from ats.datagen.text import find_keywords
from ats.models import ARCHETYPES, PII, SHORTLIST_ARCHETYPES, Archetype

# Fixed "today" so output never depends on the wall clock.
REFERENCE_YEAR = 2026
REFERENCE_MONTH = 10

DEFAULT_COUNTS: dict[str, int] = {
    "clear_fit": 6,
    "hidden_gem": 6,
    "keyword_stuffer": 6,
    "near_miss": 6,
    "career_changer": 4,
    "filler": 22,
}

YEARS_RANGE: dict[str, tuple[int, int]] = {
    "clear_fit": (6, 14),
    "hidden_gem": (6, 14),
    "near_miss": (5, 12),
    "career_changer": (5, 15),
    "keyword_stuffer": (2, 8),
    "filler": (2, 20),
}

ProofSource = Literal["jd_phrasings", "paraphrases", "transferable"]
FactVerdict = Literal["met", "partial", "missing"]

# Archetypes whose LLM prose may not use any JD keyword (their proof bullets carry the signal).
ALL_KEYWORDS_FORBIDDEN = frozenset({"hidden_gem", "career_changer", "keyword_stuffer", "filler"})


class ProofFact(BaseModel):
    requirement_id: str
    verdict: Literal["met", "partial"]
    source: ProofSource
    text: str
    entry_index: int


class RoleFact(BaseModel):
    title: str
    start: str
    end: str


class FactSheet(BaseModel):
    candidate_id: str
    job_id: str
    archetype: Archetype
    field: str
    industry: str
    years_experience: int
    roles: list[RoleFact]
    verdicts: dict[str, FactVerdict]
    proofs: list[ProofFact]
    forbidden_keywords: list[str]
    stuffed_keywords: list[str]
    pii: PII

    @property
    def expected(self) -> Literal["shortlist", "reject"]:
        return "shortlist" if self.archetype in SHORTLIST_ARCHETYPES else "reject"

    def proof_for(self, requirement_id: str) -> ProofFact | None:
        for p in self.proofs:
            if p.requirement_id == requirement_id:
                return p
        return None


def resolve_counts(overrides: dict[str, int] | None = None) -> dict[str, int]:
    counts = dict(DEFAULT_COUNTS)
    for key, value in (overrides or {}).items():
        if key not in counts:
            raise ValueError(f"unknown archetype {key!r}")
        if value < 0:
            raise ValueError(f"count for {key} must be >= 0")
        counts[key] = value
    return counts


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _fmt_month(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _build_roles(rng: random.Random, titles: list[str], years: int, n_roles: int) -> list[RoleFact]:
    total = years * 12
    min_len = 6
    weights = [rng.random() + 0.2 for _ in range(n_roles)]
    spare = total - min_len * n_roles
    durations = [min_len + int(spare * w / sum(weights)) for w in weights]
    picked = (
        rng.sample(titles, n_roles)
        if len(titles) >= n_roles
        else [rng.choice(titles) for _ in range(n_roles)]
    )
    roles: list[RoleFact] = []
    end = _month_index(REFERENCE_YEAR, REFERENCE_MONTH)
    for i, (title, months) in enumerate(zip(picked, durations, strict=True)):
        start = end - months + 1
        roles.append(
            RoleFact(
                title=title, start=_fmt_month(start), end="Present" if i == 0 else _fmt_month(end)
            )
        )
        end = start - 1
    return roles


def _pick_transferable(rng: random.Random, req: RequirementSeed, field: str) -> str:
    """Prefer bullets that fit the candidate's field, then field-neutral ones, then any."""
    own = FIELDS[field].markers
    others = {m for name in CHANGER_FIELDS if name != field for m in FIELDS[name].markers} - set(
        own
    )

    def has_marker(text: str, markers) -> bool:
        return bool(find_keywords(text, tuple(markers)))

    matched = [t for t in req.transferable if has_marker(t, own)]
    neutral = [t for t in req.transferable if not has_marker(t, others)]
    return rng.choice(matched or neutral or req.transferable)


def _make_pii(fake: Faker, rng: random.Random, years: int, used_names: set[str]) -> PII:
    while True:
        first, last = fake.first_name(), fake.last_name()
        name = f"{first} {last}"
        if name not in used_names:
            used_names.add(name)
            break
    local = re.sub(r"[^a-z.]", "", f"{first}.{last}".lower())
    email = f"{local}@{fake.free_email_domain()}"
    phone = fake.numerify("(%##) %##-####")
    address = fake.address().replace("\n", ", ")
    birth_year = REFERENCE_YEAR - (22 + years + rng.randint(0, 6))
    dob = fake.date_between_dates(date(birth_year, 1, 1), date(birth_year, 12, 31)).isoformat()
    return PII(name=name, email=email, phone=phone, address=address, dob=dob)


def _decide_verdicts(
    rng: random.Random, jd: JDSeed, archetype: str
) -> dict[str, tuple[FactVerdict, ProofSource | None]]:
    musts = [r for r in jd.requirements if r.kind == "must"]
    nices = [r for r in jd.requirements if r.kind == "nice"]
    out: dict[str, tuple[FactVerdict, ProofSource | None]] = {
        r.id: ("missing", None) for r in jd.requirements
    }

    if archetype == "clear_fit":
        for r in musts:
            out[r.id] = ("met", "jd_phrasings")
        if nices:
            for r in rng.sample(nices, rng.randint(1, len(nices))):
                out[r.id] = ("met", "jd_phrasings")
    elif archetype == "hidden_gem":
        for r in musts:
            out[r.id] = ("met", "paraphrases")
        for r in nices:
            if rng.random() < 0.5:
                out[r.id] = ("met", "paraphrases")
    elif archetype == "near_miss":
        lacking = rng.choice(musts)
        for r in musts:
            if r is not lacking:
                out[r.id] = ("met", rng.choice(("jd_phrasings", "paraphrases")))
        for r in nices:
            if rng.random() < 0.5:
                out[r.id] = ("met", rng.choice(("jd_phrasings", "paraphrases")))
    elif archetype == "career_changer":
        for r in musts:
            if rng.random() < 0.25:
                out[r.id] = ("met", "paraphrases")
            else:
                out[r.id] = ("partial", "transferable")
        if musts and all(out[r.id][0] == "met" for r in musts):
            out[rng.choice(musts).id] = ("partial", "transferable")
        for r in nices:
            if rng.random() < 0.3:
                out[r.id] = ("partial", "transferable")
    # keyword_stuffer and filler: everything stays missing.
    return out


def _build_one(
    rng: random.Random,
    fake: Faker,
    jd: JDSeed,
    archetype: str,
    candidate_id: str,
    used_names: set[str],
) -> FactSheet:
    if archetype in ("clear_fit", "hidden_gem", "near_miss"):
        field = jd.field
    elif archetype == "career_changer":
        field = rng.choice(CHANGER_FIELDS)
    elif archetype == "keyword_stuffer":
        field = rng.choice(STUFFER_FIELDS)
    else:
        field = rng.choice(FILLER_FIELDS)
    profile = FIELDS[field]

    decided = _decide_verdicts(rng, jd, archetype)
    verdicts = {rid: v for rid, (v, _) in decided.items()}

    if archetype in ALL_KEYWORDS_FORBIDDEN:
        forbidden = jd.all_keywords
    else:
        forbidden = list(
            dict.fromkeys(
                kw for r in jd.requirements if verdicts[r.id] == "missing" for kw in r.keywords
            )
        )

    lo, hi = YEARS_RANGE[archetype]
    years = rng.randint(lo, hi)
    n_roles = rng.randint(2, min(4, years))
    titles = [t for t in profile.titles if not find_keywords(t, forbidden)] or list(profile.titles)
    roles = _build_roles(rng, titles, years, n_roles)

    proofs: list[ProofFact] = []
    for req in jd.requirements:
        verdict, source = decided[req.id]
        if source is None:
            continue
        if source == "transferable":
            text = _pick_transferable(rng, req, field)
        else:
            text = rng.choice(getattr(req, source))
        proofs.append(
            ProofFact(
                requirement_id=req.id,
                verdict=verdict,  # type: ignore[arg-type]
                source=source,
                text=text,
                entry_index=rng.randrange(n_roles),
            )
        )

    stuffed: list[str] = []
    if archetype == "keyword_stuffer":
        stuffed = list(jd.all_keywords)
        rng.shuffle(stuffed)

    pii = _make_pii(fake, rng, years, used_names)
    return FactSheet(
        candidate_id=candidate_id,
        job_id=jd.job_id,
        archetype=archetype,  # type: ignore[arg-type]
        field=field,
        industry=profile.industry,
        years_experience=years,
        roles=roles,
        verdicts=verdicts,
        proofs=proofs,
        forbidden_keywords=forbidden,
        stuffed_keywords=stuffed,
        pii=pii,
    )


def build_fact_sheets(seeds: list[JDSeed], counts: dict[str, int], seed: int) -> list[FactSheet]:
    """Assign archetypes (shuffled so IDs reveal nothing); build one fact sheet per candidate."""
    rng = random.Random(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)
    used_names: set[str] = set()
    sheets: list[FactSheet] = []
    next_id = 1
    for jd in seeds:
        if counts.get("near_miss") and not any(r.kind == "must" for r in jd.requirements):
            raise ValueError(f"{jd.job_id}: near_miss needs at least one must-have requirement")
        plan = [a for a in ARCHETYPES for _ in range(counts.get(a, 0))]
        rng.shuffle(plan)
        for archetype in plan:
            sheets.append(_build_one(rng, fake, jd, archetype, f"cand_{next_id:04d}", used_names))
            next_id += 1
    return sheets
