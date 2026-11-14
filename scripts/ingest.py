"""Build the vector store from the generated dataset (SPEC §6.2).

Examples:
    uv run python scripts/ingest.py --offline
    EMBEDDING_PROVIDER=openai EMBEDDING_MODEL=... EMBEDDING_API_KEY=... \
        uv run python scripts/ingest.py

--offline uses the built-in hash embedder (no API key, no network). The server must then run with
EMBEDDING_PROVIDER=hash, because queries have to be embedded the same way as the index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import ats  # noqa: F401
except ImportError:  # allow running without installing the package
    sys.path.insert(0, str(ROOT / "src"))

from ats.config import Settings  # noqa: E402
from ats.data import DataError  # noqa: E402
from ats.rag.embeddings import (  # noqa: E402
    EmbeddingError,
    HashEmbedder,
    embedder_from_settings,
)
from ats.rag.ingest import ingest  # noqa: E402
from ats.rag.store import ResumeStore, StoreError  # noqa: E402


def _anchor(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=Path, help="generated dataset (default: DATA_DIR)")
    p.add_argument("--chroma-dir", type=Path, help="vector store folder (default: CHROMA_DIR)")
    p.add_argument("--offline", action="store_true", help="hash embedder; no API key or network")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    data_dir = _anchor(args.data_dir or settings.data_dir)
    chroma_dir = _anchor(args.chroma_dir or settings.chroma_dir)
    try:
        embedder = HashEmbedder() if args.offline else embedder_from_settings(settings)
        report = ingest(data_dir, ResumeStore(chroma_dir, embedder))
    except (DataError, EmbeddingError, StoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Indexed {report.n_resumes} resumes ({report.n_chunks} chunks, largest "
        f"~{report.largest_chunk_tokens} tokens) for {report.n_jobs} job(s) into {chroma_dir} "
        f"(embedder: {report.embedder})"
    )
    for candidate_id, kinds in report.redactions.items():
        print(f"WARNING: redacted {', '.join(kinds)} from {candidate_id}", file=sys.stderr)
    if report.missing_proofs is None:
        print("No answer key found; skipped the proof-line check.")
    elif report.missing_proofs:
        for problem in report.missing_proofs:
            print(f"error: {problem}", file=sys.stderr)
        print(
            f"error: {len(report.missing_proofs)} answer-key proof line(s) are not in the index "
            "word for word, so quote checks and the eval would be wrong.",
            file=sys.stderr,
        )
        return 1
    else:
        print("Every answer-key proof line is in the index word for word.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
