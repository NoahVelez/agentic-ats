"""Checks against the real bge-small model. Skipped by default: run with ``-m slow``.

The model (about 67 MB) is downloaded on first use, into ``EMBEDDING_CACHE_DIR`` if that is set.
"""

from __future__ import annotations

import math
import os
import statistics
from pathlib import Path

import pytest

from ats.datagen.io import load_seed
from ats.datagen.templates import FIELDS, FILLER_FIELDS
from ats.models import Chunk
from ats.rag.embeddings import BGE_QUERY_PREFIX, FastEmbedEmbedder
from ats.rag.store import ResumeStore

from ..conftest import SEED_PATH

pytestmark = pytest.mark.slow


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    cache = os.environ.get("EMBEDDING_CACHE_DIR")
    return FastEmbedEmbedder(cache_dir=Path(cache) if cache else None)


def test_real_model_vectors(embedder: FastEmbedEmbedder) -> None:
    [doc] = embedder.embed_documents(["people management"])
    query = embedder.embed_query("people management")
    assert len(doc) == len(query) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in doc)), 1.0, abs_tol=1e-5)
    assert math.isclose(math.sqrt(sum(x * x for x in query)), 1.0, abs_tol=1e-5)
    # The instruction is on the query only, so the same words give two different vectors.
    assert cosine(doc, query) < 0.999
    assert query == embedder.embed_documents([BGE_QUERY_PREFIX + "people management"])[0]


def test_index_records_the_model_and_its_vector_size(embedder: FastEmbedEmbedder, tmp_path) -> None:
    store = ResumeStore(tmp_path, embedder)
    chunk = Chunk(
        chunk_id="c1:experience:0",
        candidate_id="c1",
        section="experience",
        text="Led a 5-person analytics group, ran weekly 1:1s and quarterly reviews",
    )
    store.rebuild([chunk], {"c1": ["j1"]})
    assert store.manifest() == {"embedder": "fastembed:BAAI/bge-small-en-v1.5", "dim": 384}
    [(found, score)] = store.search("people management experience", 1)
    assert found == chunk and 0.4 < score < 1.0


def test_hidden_gem_paraphrases_outscore_unrelated_bullets(embedder: FastEmbedEmbedder) -> None:
    """Each requirement is the query. Its paraphrases (no JD keywords) are compared with bullets
    from the unrelated filler careers.

    What holds, and is asserted: every paraphrase beats the typical unrelated bullet, paraphrases
    win at least 90% of head-to-head comparisons, and each requirement's paraphrases beat unrelated
    bullets on average by a clear margin.

    What does NOT hold, and is deliberately not asserted: a paraphrase beating every unrelated
    bullet. When this was written only 9 of 22 did. A single threshold on one chunk's score cannot
    cleanly separate hidden gems from unrelated text with this model.
    """
    seed = load_seed(SEED_PATH)
    unrelated = [bullet for name in FILLER_FIELDS for bullet in FIELDS[name].bullets]
    unrelated_vectors = embedder.embed_documents(unrelated)

    win_rates: list[float] = []
    for req in seed.requirements:
        query = embedder.embed_query(req.text)
        noise = [cosine(query, v) for v in unrelated_vectors]
        scores = [cosine(query, v) for v in embedder.embed_documents(req.paraphrases)]
        assert statistics.mean(scores) - statistics.mean(noise) > 0.05, req.id
        for bullet, score in zip(req.paraphrases, scores, strict=True):
            assert score > statistics.median(noise), (req.id, bullet)
            win_rates.append(sum(score > n for n in noise) / len(noise))
    assert statistics.mean(win_rates) >= 0.90
