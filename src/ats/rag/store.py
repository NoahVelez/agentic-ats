"""ChromaDB-backed resume store (SPEC §6.2).

Two collections:
- ``resumes``: one row per chunk, with metadata ``{candidate_id, section, job_ids, order}``.
- ``resume_docs``: one row per candidate (all of that candidate's chunks joined), used by the
  whole-resume baselines.

Scores are cosine similarity clamped to [0, 1] (``1 - cosine distance``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from ats.models import Chunk, Section
from ats.rag.embeddings import Embedder

CHUNKS_COLLECTION = "resumes"
DOCS_COLLECTION = "resume_docs"
_BATCH = 1000


class StoreError(RuntimeError):
    pass


class StoreNotReady(StoreError):
    pass


class EmbedderMismatch(StoreError):
    pass


def similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _and(conditions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class ResumeStore:
    def __init__(self, chroma_dir: Path, embedder: Embedder) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.embedder = embedder
        self._client: Any = None
        self._lock = threading.Lock()

    def _open(self, create: bool) -> Any:
        with self._lock:
            if self._client is None:
                if not create and not self.chroma_dir.is_dir():
                    raise StoreNotReady(
                        f"no vector store at {self.chroma_dir}; run scripts/ingest.py first"
                    )
                self.chroma_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.chroma_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            return self._client

    def _collection(self, name: str) -> Any:
        client = self._open(create=False)
        try:
            col = client.get_collection(name, embedding_function=None)
        except NotFoundError as exc:
            raise StoreNotReady(
                f"vector store at {self.chroma_dir} has no data; run scripts/ingest.py first"
            ) from exc
        indexed = (col.metadata or {}).get("embedder")
        if indexed != self.embedder.name:
            raise EmbedderMismatch(
                f"the index was built with embedder {indexed!r} but {self.embedder.name!r} is "
                "configured. Re-run scripts/ingest.py, or set EMBEDDING_PROVIDER/EMBEDDING_MODEL "
                "to match the index."
            )
        return col

    # --- write side ---------------------------------------------------------------------------

    def rebuild(self, chunks: list[Chunk], job_ids: dict[str, list[str]]) -> None:
        """Replace the whole index. Everything is embedded first, so a failed embedding call
        leaves the previous index in place."""
        if not chunks:
            raise ValueError("no chunks to index")
        by_candidate: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_candidate.setdefault(chunk.candidate_id, []).append(chunk)
        missing = sorted(set(by_candidate) - set(job_ids))
        if missing:
            raise ValueError(f"no job_ids given for candidates {missing[:5]}")

        doc_ids = sorted(by_candidate)
        doc_texts = ["\n".join(c.text for c in by_candidate[cid]) for cid in doc_ids]
        chunk_vectors = self.embedder.embed([c.text for c in chunks])
        doc_vectors = self.embedder.embed(doc_texts)

        order = {c.chunk_id: i for cid in doc_ids for i, c in enumerate(by_candidate[cid])}
        client = self._open(create=True)
        meta = {"hnsw:space": "cosine", "embedder": self.embedder.name}
        for name in (CHUNKS_COLLECTION, DOCS_COLLECTION):
            try:
                client.delete_collection(name)
            except NotFoundError:
                pass
        chunk_col = client.create_collection(
            CHUNKS_COLLECTION, metadata=meta, embedding_function=None
        )
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            chunk_col.add(
                ids=[c.chunk_id for c in batch],
                embeddings=chunk_vectors[i : i + _BATCH],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "candidate_id": c.candidate_id,
                        "section": c.section,
                        "job_ids": list(job_ids[c.candidate_id]),
                        "order": order[c.chunk_id],
                    }
                    for c in batch
                ],
            )
        doc_col = client.create_collection(DOCS_COLLECTION, metadata=meta, embedding_function=None)
        for i in range(0, len(doc_ids), _BATCH):
            doc_col.add(
                ids=doc_ids[i : i + _BATCH],
                embeddings=doc_vectors[i : i + _BATCH],
                documents=doc_texts[i : i + _BATCH],
                metadatas=[{"job_ids": list(job_ids[cid])} for cid in doc_ids[i : i + _BATCH]],
            )

    # --- read side ----------------------------------------------------------------------------

    def is_ready(self) -> bool:
        try:
            return self._collection(CHUNKS_COLLECTION).count() > 0
        except StoreError:
            return False

    def counts(self) -> dict[str, int]:
        return {
            "chunks": self._collection(CHUNKS_COLLECTION).count(),
            "candidates": self._collection(DOCS_COLLECTION).count(),
        }

    def search(
        self,
        query: str,
        k: int,
        *,
        section: Section | None = None,
        candidate_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        col = self._collection(CHUNKS_COLLECTION)
        conditions: list[dict[str, Any]] = []
        if candidate_id is not None:
            conditions.append({"candidate_id": candidate_id})
        if section is not None:
            conditions.append({"section": section})
        res = col.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=k,
            where=_and(conditions),
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            (
                Chunk(chunk_id=cid, candidate_id=m["candidate_id"], section=m["section"], text=doc),
                similarity(dist),
            )
            for cid, doc, m, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
                strict=True,
            )
        ]
        return sorted(hits, key=lambda hit: (-hit[1], hit[0].chunk_id))

    def candidate_chunks(
        self, candidate_id: str, sections: list[Section] | None = None
    ) -> list[Chunk]:
        """All chunks of one candidate, in resume order."""
        col = self._collection(CHUNKS_COLLECTION)
        conditions: list[dict[str, Any]] = [{"candidate_id": candidate_id}]
        if sections:
            conditions.append({"section": {"$in": list(sections)}})
        res = col.get(where=_and(conditions), include=["documents", "metadatas"])
        rows = sorted(
            zip(res["ids"], res["documents"], res["metadatas"], strict=True),
            key=lambda row: row[2]["order"],
        )
        return [
            Chunk(chunk_id=cid, candidate_id=m["candidate_id"], section=m["section"], text=doc)
            for cid, doc, m in rows
        ]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        res = self._collection(CHUNKS_COLLECTION).get(
            ids=[chunk_id], include=["documents", "metadatas"]
        )
        if not res["ids"]:
            return None
        meta = res["metadatas"][0]
        return Chunk(
            chunk_id=res["ids"][0],
            candidate_id=meta["candidate_id"],
            section=meta["section"],
            text=res["documents"][0],
        )

    def has_candidate(self, candidate_id: str) -> bool:
        return bool(self._collection(DOCS_COLLECTION).get(ids=[candidate_id], include=[])["ids"])

    def documents(self, job_id: str) -> dict[str, str]:
        """Whole-resume text (redacted) for every candidate in a job's pool."""
        res = self._collection(DOCS_COLLECTION).get(
            where={"job_ids": {"$contains": job_id}}, include=["documents"]
        )
        return dict(sorted(zip(res["ids"], res["documents"], strict=True)))

    def rank_by_embedding(self, text: str, job_id: str) -> list[tuple[str, float]]:
        """Every candidate in a job's pool, ranked by whole-resume similarity to ``text``."""
        col = self._collection(DOCS_COLLECTION)
        total = col.count()
        if total == 0:
            return []
        res = col.query(
            query_embeddings=self.embedder.embed([text]),
            n_results=total,
            where={"job_ids": {"$contains": job_id}},
            include=["distances"],
        )
        scored = [
            (cid, similarity(dist))
            for cid, dist in zip(res["ids"][0], res["distances"][0], strict=True)
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0]))
