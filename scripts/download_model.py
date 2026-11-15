"""Download the embedding model ahead of time, so nothing is fetched when the server starts.

Run at Docker build time. Reads EMBEDDING_PROVIDER, EMBEDDING_MODEL and EMBEDDING_CACHE_DIR like
the server does, and does nothing for providers that have no local model.

    uv run python scripts/download_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import ats  # noqa: F401
except ImportError:  # allow running without installing the package
    sys.path.insert(0, str(ROOT / "src"))

from ats.config import Settings  # noqa: E402
from ats.rag.embeddings import (  # noqa: E402
    EmbeddingError,
    FastEmbedEmbedder,
    embedder_from_settings,
)


def main() -> int:
    settings = Settings()
    try:
        embedder = embedder_from_settings(settings)
        if not isinstance(embedder, FastEmbedEmbedder):
            print(f"{embedder.name} has no local model to download.")
            return 0
        dim = len(embedder.embed_documents(["warm-up"])[0])
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    where = settings.embedding_cache_dir or "fastembed's default cache folder"
    print(f"{embedder.name} is ready ({dim} dimensions) in {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
