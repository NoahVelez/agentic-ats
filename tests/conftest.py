from __future__ import annotations

from pathlib import Path

import pytest

from ats.data import load_answer_key, load_resumes
from ats.datagen.io import load_seed, write_outputs
from ats.datagen.pipeline import GenerationConfig, generate
from ats.datagen.render import OfflineRenderer
from ats.models import JobAnswerKey, Resume
from ats.rag.embeddings import HashEmbedder
from ats.rag.ingest import IngestReport, ingest
from ats.rag.store import ResumeStore

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "jd_seeds" / "analytics_manager.yaml"
JOB_ID = "jd_analytics_mgr"


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> Path:
    """The default 50-resume pool, generated offline with seed 42."""
    out = tmp_path_factory.mktemp("dataset")
    result = generate([load_seed(SEED_PATH)], OfflineRenderer(42), GenerationConfig(seed=42))
    write_outputs(out, result.jobs, result.resumes, result.answer_key, result.report)
    return out


@pytest.fixture(scope="session")
def resumes(dataset: Path) -> list[Resume]:
    return load_resumes(dataset)


@pytest.fixture(scope="session")
def answer_key(dataset: Path) -> JobAnswerKey:
    return load_answer_key(dataset).jobs[0]


@pytest.fixture(scope="session")
def indexed(dataset: Path, tmp_path_factory) -> tuple[Path, IngestReport]:
    """The dataset ingested with the offline hash embedder: (chroma_dir, report)."""
    chroma_dir = tmp_path_factory.mktemp("chroma")
    report = ingest(dataset, ResumeStore(chroma_dir, HashEmbedder()))
    return chroma_dir, report


@pytest.fixture(scope="session")
def store(indexed: tuple[Path, IngestReport]) -> ResumeStore:
    return ResumeStore(indexed[0], HashEmbedder())
