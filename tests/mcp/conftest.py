from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastmcp import Client

from ats.config import Settings
from ats.mcp_server import Runtime, create_mcp


@pytest.fixture(scope="session")
def jobs_only_dir(dataset: Path, tmp_path_factory) -> Path:
    """A data folder holding only jobs/. The server gets no resumes file and no answer key, so
    a tool that tried to read either would fail."""
    out = tmp_path_factory.mktemp("jobs_only")
    shutil.copytree(dataset / "jobs", out / "jobs")
    return out


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{"embedding_provider": "hash", **overrides})


@pytest.fixture
def runtime(indexed, jobs_only_dir: Path, tmp_path: Path) -> Runtime:
    return Runtime(
        make_settings(
            data_dir=jobs_only_dir,
            chroma_dir=indexed[0],
            runs_dir=tmp_path / "runs",
            strong_hit=0.3,
        )
    )


@pytest.fixture
async def client(runtime: Runtime):
    async with Client(create_mcp(runtime)) as c:
        yield c
