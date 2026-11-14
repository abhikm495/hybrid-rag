from rag_pipeline.chunking.models import Chunk

def chunk_to_chroma_metadata(chunk: Chunk) -> dict:
    m = chunk.metadata
    meta = {
        "source_file": m.source_file,
        "section_heading": m.section_heading,
        "chunk_index": m.chunk_index,
        "chunking_strategy": m.chunking_strategy.value,  # enum → string
        "char_count": m.char_count,
        "document_id": m.document_id,
    }
    if m.page_number is not None:
        meta["page_number"] = m.page_number
    return meta