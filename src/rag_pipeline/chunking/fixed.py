from rag_pipeline.chunking.models import Chunk, ChunkMetadata, ChunkingConfig, ChunkingStrategy
from rag_pipeline.ingestion.models import ProcessedDocument, TextBlock



class FixedSizeChunker:
    def _make_chunk(self, doc: ProcessedDocument, piece: str, index: int) -> Chunk:
        meta = ChunkMetadata(
        source_file=doc.metadata.source_file,
        section_heading=doc.metadata.section_heading,
        page_number=doc.metadata.page_number,
        chunk_index=index,
        chunking_strategy=ChunkingStrategy.FIXED,
        char_count=len(piece),
        document_id=doc.document_id,
    )
        return Chunk(
        chunk_id=f"{doc.document_id}:fixed:{index}",
        text=piece,
        metadata=meta,
    )
    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(self, doc: ProcessedDocument, _blocks: list[TextBlock]) -> list[Chunk]:
        text = doc.plaintext
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = max(1, size - overlap)  # guard: avoid infinite loop if overlap >= size

        chunks: list[Chunk] = []
        index = 0
        start = 0

        while start < len(text):
            piece = text[start : start + size]
            if not piece.strip():
                break
            chunks.append(self._make_chunk(doc, piece, index))
            index += 1
            start += step

        return chunks

