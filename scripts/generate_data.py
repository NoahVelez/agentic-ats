"""Generate synthetic JDs, resumes, and the answer key (SPEC §6.1).

Examples:
    uv run python scripts/generate_data.py --offline
    DATAGEN_LLM_PROVIDER=openai DATAGEN_LLM_MODEL=... DATAGEN_LLM_API_KEY=... \
        uv run python scripts/generate_data.py --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import ats  # noqa: F401
except ImportError:  # allow running without installing the package
    sys.path.insert(0, str(ROOT / "src"))

from ats.datagen.facts import DEFAULT_COUNTS, resolve_counts  # noqa: E402
from ats.datagen.io import load_seeds, write_outputs  # noqa: E402
from ats.datagen.llm import DEFAULT_CACHE_DIR, ConfigError, client_from_env  # noqa: E402
from ats.datagen.pipeline import GenerationConfig, GenerationError, generate  # noqa: E402
from ats.datagen.render import LLMRenderer, OfflineRenderer  # noqa: E402
from ats.datagen.validate import DEFAULT_JACCARD_THRESHOLD  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--seeds",
        nargs="+",
        type=Path,
        default=[ROOT / "data" / "jd_seeds"],
        help="JD seed YAML files or directories (default: data/jd_seeds)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "generated",
        help="output directory (default: data/generated, gitignored)",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    p.add_argument("--offline", action="store_true", help="template renderer, no LLM calls")
    for name, default in DEFAULT_COUNTS.items():
        p.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=int,
            default=default,
            help=f"{name} resumes per JD (default: {default})",
        )
    p.add_argument(
        "--jaccard-threshold",
        type=float,
        default=DEFAULT_JACCARD_THRESHOLD,
        help="max hidden_gem proof/requirement token overlap (default: %(default)s)",
    )
    p.add_argument("--max-attempts", type=int, default=3, help="render attempts per candidate")
    p.add_argument("--workers", type=int, default=4, help="concurrent render workers")
    p.add_argument("--cache-dir", type=Path, default=ROOT / DEFAULT_CACHE_DIR)
    p.add_argument("--no-cache", action="store_true", help="disable the LLM response cache")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        counts = resolve_counts({k: getattr(args, k) for k in DEFAULT_COUNTS})
        seeds = load_seeds(args.seeds)
        if args.offline:
            renderer = OfflineRenderer(args.seed)
        else:
            client = client_from_env(cache_dir=None if args.no_cache else args.cache_dir)
            renderer = LLMRenderer(client)
        config = GenerationConfig(
            seed=args.seed,
            counts=counts,
            jaccard_threshold=args.jaccard_threshold,
            max_attempts=args.max_attempts,
            workers=args.workers,
        )
        result = generate(seeds, renderer, config)
    except (ConfigError, GenerationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_outputs(args.out, result.jobs, result.resumes, result.answer_key, result.report)
    print(
        f"Wrote {len(result.jobs)} job(s), {len(result.resumes)} resumes, and the answer key "
        f"to {args.out} (generator: {renderer.name})"
    )
    for job_id, job in result.report["bm25"].items():
        print(f"\nBM25 ranks for {job_id} (pool {job['pool_size']}, top-N = {job['top_n']}):")
        print(
            "  hidden_gem:      "
            + ", ".join(f"{c}={r}" for c, r in job["hidden_gem_ranks"].items())
        )
        print(
            "  keyword_stuffer: "
            + ", ".join(f"{c}={r}" for c, r in job["keyword_stuffer_ranks"].items())
        )
    for warning in result.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
