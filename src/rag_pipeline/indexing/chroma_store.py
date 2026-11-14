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