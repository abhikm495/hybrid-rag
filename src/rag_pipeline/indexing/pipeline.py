from pathlib import Path

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.embeddings import embed_texts
from rag_pipeline.indexing.bm25_store import BM25ChunkStore
from rag_pipeline.indexing.chroma_store import ChromaChunkStore
from rag_pipeline.indexing.dedup import (
    DEDUP_SIMILARITY_THRESHOLD,
    append_dedup_log,
    filter_near_duplicates,
)
from rag_pipeline.indexing.validate import validate_chunks_for_indexing


def index_chunks(
    chunks: list[Chunk],
    *,
    data_dir: Path,
    embedding_model: str = "text-embedding-3-small",
    collection_name: str = "chunks_v1",
    dedup_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> dict:
    validate_chunks_for_indexing(chunks)

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, model=embedding_model)

    chroma = ChromaChunkStore(
        persist_dir=data_dir / "index" / "chroma",
        collection_name=collection_name,
    )
    kept_chunks, _kept_embeddings, decisions = filter_near_duplicates(
        chunks,
        embeddings,
        chroma,
        threshold=dedup_threshold,
        upsert_kept=True,
    )

    append_dedup_log(decisions, data_dir / "index" / "dedup_log.jsonl")

    bm25 = BM25ChunkStore()
    if kept_chunks:
        bm25.merge_and_save(data_dir / "index" / "bm25", kept_chunks)

    skipped = len(chunks) - len(kept_chunks)
    return {
        "indexed": len(kept_chunks),
        "skipped_duplicates": skipped,
        "chroma_count": chroma.count(),
        "embedding_model": embedding_model,
        "collection_name": collection_name,
        "dedup_threshold": dedup_threshold,
    }