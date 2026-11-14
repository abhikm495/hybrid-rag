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