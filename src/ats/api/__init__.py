"""FastAPI routes under ``/api`` (SPEC §8). Only ``/health`` exists so far."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

if TYPE_CHECKING:
    from ats.mcp_server import Runtime


def create_router(runtime: Runtime) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, object]:
        """Liveness, plus whether the vector store has been built and matches the embedder."""
        return {"status": "ok", "vector_store_loaded": runtime.store_ready()}

    return router
