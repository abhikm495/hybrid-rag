from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.indexing.metadata import chunk_to_chroma_metadata  # your helper

class ChromaChunkStore:
    def __init__(self, persist_dir: Path, collection_name: str = "chunks_v1"):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # matches cosine similarity retrieval
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[chunk_to_chroma_metadata(c) for c in chunks],
        )

    def count(self) -> int:
        return self._collection.count()

    def get_metadata_by_ids(
        self,
        chunk_ids: list[str],
    ) -> dict[str, dict[str, str | int | float | bool]]:
        """Return Chroma metadata keyed by chunk_id (unknown ids are omitted)."""
        if not chunk_ids:
            return {}

        result = self._collection.get(ids=chunk_ids, include=["metadatas"])
        ids: list[str] = result.get("ids") or []
        metadatas: list[dict | None] = result.get("metadatas") or []

        out: dict[str, dict[str, str | int | float | bool]] = {}
        for chunk_id, meta in zip(ids, metadatas, strict=True):
            if meta is not None:
                out[chunk_id] = meta
        return out

    def find_nearest_neighbor(
        self,
        query_embedding: list[float],
        *,
        exclude_ids: set[str] | None = None,
    ) -> tuple[str, float] | None:
        """Return (chunk_id, cosine similarity) to the nearest indexed chunk, if any."""
        n_indexed = self._collection.count()
        if n_indexed == 0:
            return None

        exclude = exclude_ids or set()
        n_results = min(max(10, len(exclude) + 1), n_indexed)

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances"],
        )

        ids: list[str] = result["ids"][0]
        distances: list[float] = result["distances"][0]

        for chunk_id, distance in zip(ids, distances):
            if chunk_id in exclude:
                continue
            # Chroma cosine space: distance ≈ 1 - similarity (normalized embeddings).
            similarity = 1.0 - float(distance)
            return chunk_id, similarity

        return None

    def search_by_embedding(
    self,
    query_embedding: list[float],
    *,
    k: int = 10,
) -> list[tuple[str, str, dict, float]]:
        """ Returns list of (chunk_id, text, metadata, similarity), best first. """
        n_indexed = self._collection.count()
        if n_indexed == 0:
            return []

        n_results = min(k, n_indexed)

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids: list[str] = result["ids"][0]
        documents: list[str] = result["documents"][0]
        metadatas: list[dict] = result["metadatas"][0]
        distances: list[float] = result["distances"][0]

        hits: list[tuple[str, str, dict, float]] = []
        for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - float(distance)
            hits.append((chunk_id, doc, meta, similarity))
        return hits