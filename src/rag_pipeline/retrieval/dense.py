from __future__ import annotations

from pathlib import Path

from rag_pipeline.embeddings import embed_texts
from rag_pipeline.indexing.chroma_store import ChromaChunkStore
from rag_pipeline.retrieval.models import RetrievedChunk


class DenseRetriever:
    def __init__(
        self,
        *,
        chroma_persist_dir: Path,
        collection_name: str = "chunks_v1",
        embedding_model: str = "text-embedding-3-small",
        default_k: int = 10,
    ) -> None:
        self._embedding_model = embedding_model
        self._default_k = default_k
        self._store = ChromaChunkStore(
            persist_dir=chroma_persist_dir,
            collection_name=collection_name,
        )

    def retrieve(self, question: str, *, k: int | None = None) -> list[RetrievedChunk]:
        k = k if k is not None else self._default_k
        question = question.strip()
        if not question:
            return []

        query_vector = embed_texts([question], model=self._embedding_model)[0]
        raw_hits = self._store.search_by_embedding(query_vector, k=k)

        results: list[RetrievedChunk] = []
        for rank, (chunk_id, text, metadata, score) in enumerate(raw_hits, start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    metadata=metadata,
                    rank=rank,
                )
            )
        return results