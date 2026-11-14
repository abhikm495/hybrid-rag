from rag_pipeline.chunking.models import Chunk

def validate_chunks_for_indexing(chunks: list[Chunk]) -> None:
    if not chunks:
        raise ValueError("No chunks to index")

    seen: set[str] = set()
    for c in chunks:
        if not c.chunk_id:
            raise ValueError("chunk missing chunk_id")
        if c.chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {c.chunk_id}")
        seen.add(c.chunk_id)
        if not c.text.strip():
            raise ValueError(f"empty text for {c.chunk_id}")

