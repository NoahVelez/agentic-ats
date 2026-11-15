"""Calibrate STRONG_HIT for the configured embedding model (SPEC §4.2).

Each requirement's text is used as a search query and scored against the chunk that holds each
planted proof line in the answer key. STRONG_HIT is the 25th percentile of those scores, so three
quarters of the true proof chunks count as strong hits.

The script also reports how many candidates clear that score for each requirement, split by
whether the answer key says they have it, and an AUC: the chance that a candidate who has the
requirement outscores one who lacks it (0.5 is no better than chance, 1.0 is perfect). A threshold
is only useful if the two groups differ.

    uv run python scripts/calibrate.py
    uv run python scripts/calibrate.py --section experience   # search experience chunks only

Reads the same settings as the server (EMBEDDING_*, DATA_DIR, CHROMA_DIR). Run ingest first.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import ats  # noqa: F401
except ImportError:  # allow running without installing the package
    sys.path.insert(0, str(ROOT / "src"))

from ats.config import Settings  # noqa: E402
from ats.data import load_answer_key  # noqa: E402
from ats.models import AnswerKey, Section  # noqa: E402
from ats.rag.chunking import bullet_line  # noqa: E402
from ats.rag.embeddings import EmbeddingError, embedder_from_settings  # noqa: E402
from ats.rag.store import ResumeStore, StoreError  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, q from 0 to 100."""
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass
class RequirementReach:
    requirement_id: str
    has_it: int  # candidates the answer key marks met or partial
    has_it_reached: int  # ... of those, how many have a chunk at or above the threshold
    lacks_it: int  # candidates the answer key marks missing
    lacks_it_reached: int
    auc: float  # chance that a candidate who has it outscores one who lacks it


@dataclass
class Calibration:
    embedder: str
    q: float
    threshold: float
    section: str | None = None
    # (archetype, verdict, requirement_id, score) for every proof line
    proofs: list[tuple[str, str, str, float]] = field(default_factory=list)
    reach: list[RequirementReach] = field(default_factory=list)
    # candidates with a chunk at or above the threshold for at least one requirement
    any_strong_hit: dict[str, tuple[int, int]] = field(default_factory=dict)


def auc(has: list[float], lacks: list[float]) -> float:
    if not has or not lacks:
        return float("nan")
    wins = sum((h > x) + 0.5 * (h == x) for h in has for x in lacks)
    return wins / (len(has) * len(lacks))


def calibrate(
    store: ResumeStore, answer_key: AnswerKey, q: float = 25.0, section: Section | None = None
) -> Calibration:
    """``section`` restricts the search, and the proof lines counted, to one resume section."""
    total_chunks = store.counts()["chunks"]
    proofs: list[tuple[str, str, str, float]] = []
    best: dict[tuple[str, str], float] = {}  # (requirement_id, candidate_id) -> best chunk score
    verdicts: dict[tuple[str, str], str] = {}
    archetypes: dict[str, str] = {}
    expected: dict[str, str] = {}

    for job in answer_key.jobs:
        scores_by_req: dict[str, dict[str, float]] = {}
        texts: dict[str, str] = {}
        for req in job.requirements:
            hits = store.search(req.text, total_chunks, section=section)
            scores_by_req[req.id] = {chunk.chunk_id: score for chunk, score in hits}
            for chunk, score in hits:
                texts[chunk.chunk_id] = chunk.text
                pair = (req.id, chunk.candidate_id)
                best[pair] = max(best.get(pair, 0.0), score)
        for cand in job.candidates:
            archetypes[cand.candidate_id] = cand.archetype
            expected[cand.candidate_id] = cand.expected
            prefix = cand.candidate_id + ":"
            for item in cand.requirements:
                verdicts[(item.requirement_id, cand.candidate_id)] = item.verdict
                if not item.proof or (section and item.location.section != section):
                    continue
                wanted = bullet_line(item.proof)
                holders = [
                    cid
                    for cid, text in texts.items()
                    if cid.startswith(prefix) and wanted in text.splitlines()
                ]
                if len(holders) != 1:
                    raise StoreError(
                        f"{cand.candidate_id} {item.requirement_id}: proof line is in "
                        f"{len(holders)} chunks; re-run ingest"
                    )
                score = scores_by_req[item.requirement_id][holders[0]]
                proofs.append((cand.archetype, item.verdict, item.requirement_id, score))

    if not proofs:
        raise StoreError("the answer key has no proof lines to calibrate against")
    threshold = round(percentile([p[3] for p in proofs], q), 3)

    reach = []
    for req_id in sorted({r for r, _ in best}):
        has = [c for (r, c), v in verdicts.items() if r == req_id and v != "missing"]
        lacks = [c for (r, c), v in verdicts.items() if r == req_id and v == "missing"]
        has_scores = [best.get((req_id, c), 0.0) for c in has]
        lacks_scores = [best.get((req_id, c), 0.0) for c in lacks]
        reach.append(
            RequirementReach(
                requirement_id=req_id,
                has_it=len(has),
                has_it_reached=sum(x >= threshold for x in has_scores),
                lacks_it=len(lacks),
                lacks_it_reached=sum(x >= threshold for x in lacks_scores),
                auc=auc(has_scores, lacks_scores),
            )
        )
    any_hit: dict[str, tuple[int, int]] = {}
    for group in sorted(set(archetypes.values())):
        members = [c for c, a in archetypes.items() if a == group]
        reached = {c for (r, c), s in best.items() if s >= threshold and c in members}
        any_hit[group] = (len(reached), len(members))
    return Calibration(store.embedder.name, q, threshold, section, proofs, reach, any_hit)


def _row(label: str, values: list[float]) -> str:
    cells = [min(values), percentile(values, 10), percentile(values, 25)]
    cells += [percentile(values, 50), max(values)]
    return f"  {label:<26}{len(values):>4}" + "".join(f"{v:>8.3f}" for v in cells)


def render(result: Calibration) -> str:
    lines = [
        f"Embedder: {result.embedder}",
        f"Searching: {result.section + ' chunks only' if result.section else 'all sections'}",
        "",
        "Scores of the chunks holding planted proof lines (requirement text as the query):",
        f"  {'group':<26}{'n':>4}{'min':>8}{'p10':>8}{'p25':>8}{'median':>8}{'max':>8}",
        _row("all proof lines", [p[3] for p in result.proofs]),
    ]
    for archetype in sorted({p[0] for p in result.proofs}):
        lines.append(_row(archetype, [p[3] for p in result.proofs if p[0] == archetype]))
    for verdict in sorted({p[1] for p in result.proofs}):
        lines.append(_row(f"verdict: {verdict}", [p[3] for p in result.proofs if p[1] == verdict]))
    lines += [
        "",
        f"STRONG_HIT = {result.threshold}   (percentile {result.q:g} of all proof-line scores)",
        "",
        "Candidates with at least one chunk at or above STRONG_HIT, per requirement:",
        f"  {'req':<6}{'key says they have it':>24}{'key says they lack it':>26}{'AUC':>8}",
    ]
    for r in result.reach:
        have = f"{r.has_it_reached} of {r.has_it}"
        lack = f"{r.lacks_it_reached} of {r.lacks_it}"
        lines.append(f"  {r.requirement_id:<6}{have:>24}{lack:>26}{r.auc:>8.2f}")
    lines += ["", "Candidates with a strong hit on at least one requirement, by archetype:"]
    for group, (reached, total) in result.any_strong_hit.items():
        lines.append(f"  {group:<18}{reached:>3} of {total}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, help="generated dataset (default: DATA_DIR)")
    parser.add_argument("--chroma-dir", type=Path, help="vector store (default: CHROMA_DIR)")
    parser.add_argument(
        "--percentile", type=float, default=25.0, help="percentile of proof scores (default: 25)"
    )
    parser.add_argument(
        "--section",
        choices=["summary", "experience", "education", "skills", "projects"],
        help="search one resume section only (default: all sections)",
    )
    parser.add_argument("--offline", action="store_true", help="index was built with --offline")
    args = parser.parse_args(argv)
    settings = Settings()

    def anchor(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    try:
        if args.offline:
            from ats.rag.embeddings import HashEmbedder

            embedder = HashEmbedder()
        else:
            embedder = embedder_from_settings(settings)
        answer_key = load_answer_key(anchor(args.data_dir or settings.data_dir))
        if answer_key is None:
            print("error: no answer_key.json; run scripts/generate_data.py", file=sys.stderr)
            return 1
        store = ResumeStore(anchor(args.chroma_dir or settings.chroma_dir), embedder)
        result = calibrate(store, answer_key, args.percentile, args.section)
    except (EmbeddingError, StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
