"""Settings, read from the environment and an optional ``.env`` file (SPEC §12).

Empty values are treated as unset, so a ``.env`` copied from ``.env.example`` works before it is
filled in.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Paths (relative paths resolve against the working directory).
    data_dir: Path = Path("data/generated")
    chroma_dir: Path = Path("chroma")
    runs_dir: Path = Path("runs")

    # App LLM. Separate from the generator's DATAGEN_LLM_* variables.
    llm_provider: str = ""
    llm_api_key: SecretStr | None = None
    llm_model: str = ""

    # Embeddings: "openai" (any OpenAI-compatible endpoint) or "hash" (offline, lexical).
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str = "https://api.openai.com/v1"

    # Service
    mcp_url: str = "http://localhost:8000/mcp/"
    mcp_api_key: SecretStr | None = None
    app_api_key: SecretStr | None = None
    api_base_url: str = "http://localhost:8000"

    # Caps (SPEC §4.5)
    max_search_rounds: int = Field(default=3, ge=1)
    max_verify_rounds: int = Field(default=2, ge=0)
    max_clarify_questions: int = Field(default=3, ge=0)
    max_pool_size: int = Field(default=20, ge=1)
    run_token_budget: int = Field(default=300_000, ge=1)
    run_cost_budget_usd: float = Field(default=1.00, gt=0)
    eval_concurrency: int = Field(default=5, ge=1)
    # Calibrated per embedding model; there is no meaningful default (SPEC §4.2).
    strong_hit: float | None = Field(default=None, ge=0, le=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
