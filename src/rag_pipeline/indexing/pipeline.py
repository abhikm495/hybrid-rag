from pathlib import Path

from rag_pipeline.chunking.models import Chunk, ChunkingConfig
from rag_pipeline.embeddings import embed_texts
from rag_pipeline.indexing.bm25_store import BM25ChunkStore
from rag_pipeline.indexing.chroma_store import ChromaChunkStore
from rag_pipeline.indexing.validate import validate_chunks_for_indexing

def index_chunks(
    chunks: list[Chunk],
    *,
    data_dir: Path,
    embedding_model: str = "text-embedding-3-small",
    collection_name: str = "chunks_v1",
) -> dict:
    validate_chunks_for_indexing(chunks)

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, model=embedding_model)

  # Dense
    chroma = ChromaChunkStore(
        persist_dir=data_dir / "index" / "chroma",
        collection_name=collection_name,
    )
    chroma.upsert_chunks(chunks, embeddings)

    # Sparse — same chunk order as texts/embeddings
    bm25 = BM25ChunkStore()
    bm25.rebuild(chunks)
    bm25.save(data_dir / "index" / "bm25", chunks)

    return {
        "indexed": len(chunks),
        "chroma_count": chroma.count(),
        "embedding_model": embedding_model,
        "collection_name": collection_name,
    }