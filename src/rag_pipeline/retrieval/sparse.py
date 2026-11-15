from __future__ import annotations

from pathlib import Path

from rag_pipeline.indexing.bm25_store import BM25ChunkStore
from rag_pipeline.indexing.chroma_store import ChromaChunkStore
from rag_pipeline.retrieval.models import RetrievedChunk


class SparseRetriever:
    def __init__(
        self,
        *,
        bm25_dir: Path,
        chroma_persist_dir: Path | None = None,
        collection_name: str = "chunks_v1",
        default_k: int = 10,
    ) -> None:
        self._default_k = default_k
        self._store = BM25ChunkStore()
        self._store.load(bm25_dir)
        self._chroma: ChromaChunkStore | None = None
        if chroma_persist_dir is not None:
            self._chroma = ChromaChunkStore(
                persist_dir=chroma_persist_dir,
                collection_name=collection_name,
            )

    def retrieve(self, question: str, *, k: int | None = None) -> list[RetrievedChunk]:
        k = k if k is not None else self._default_k
        question = question.strip()
        if not question:
            return []

        raw_hits = self._store.top_k(question, k=k)

        metadata_by_id: dict[str, dict[str, str | int | float | bool]] = {}
        if self._chroma is not None and raw_hits:
            chunk_ids = [chunk_id for chunk_id, _ in raw_hits]
            metadata_by_id = self._chroma.get_metadata_by_ids(chunk_ids)

        results: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(raw_hits, start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=self._store.text_by_id(chunk_id),
                    score=score,
                    metadata=metadata_by_id.get(chunk_id, {}),
                    rank=rank,
                )
            )
        return results