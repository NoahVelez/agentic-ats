FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    EMBEDDING_PROVIDER=fastembed \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    EMBEDDING_CACHE_DIR=/app/models

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
RUN uv sync --locked --no-dev

# Fetch the embedding model into EMBEDDING_CACHE_DIR now, then forbid downloads at runtime.
RUN uv run --no-sync python scripts/download_model.py
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "python", "-m", "uvicorn", "ats.main:app", "--host", "0.0.0.0", "--port", "8000"]
