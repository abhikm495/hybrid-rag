from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.chunking.models import Chunk, ChunkMetadata, ChunkingConfig, ChunkingStrategy
from rag_pipeline.ingestion.models import ProcessedDocument, TextBlock


from collections import defaultdict
from rag_pipeline.ingestion.models import TextBlock



def build_section_text(section_blocks: list[TextBlock]) -> str:
    parts: list[str] = []
    for block in section_blocks:
        text = block.text.strip()
        if not text:
            continue
        parts.append(text)
    return "\n\n".join(parts)

def group_by_section(blocks: list[TextBlock]) -> dict[str, list[TextBlock]]:
    grouped: dict[str, list[TextBlock]] = defaultdict(list)
    for block in blocks:
        key = block.section_heading or "Root"
        grouped[key].append(block)
    return dict(grouped)

def pick_page(section_blocks: list[TextBlock]) -> int | None:
    for block in section_blocks:
        if block.page_number is not None:
            return block.page_number
    return None

class RecursiveChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def _split_recursively(self, section_text: str) -> list[str]:
        text = section_text
        if not text.strip():
            return []

        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = max(1, size - overlap)

        pieces: list[str] = []
        start = 0
        while start < len(text):
            piece = text[start : start + size]
            if not piece.strip():
                break
            pieces.append(piece)
            start += step

        return pieces if pieces else [text]

    def _make_chunk(
        self,
        doc: ProcessedDocument,
        piece: str,
        index: int,
        *,
        section_heading: str,
        page_number: int | None,
    ) -> Chunk:
        meta = ChunkMetadata(
            source_file=doc.metadata.source_file,
            section_heading=section_heading,
            page_number=page_number,
            chunk_index=index,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            char_count=len(piece),
            document_id=doc.document_id,
        )
        return Chunk(
            chunk_id=f"{doc.document_id}:recursive:{index}",
            text=piece,
            metadata=meta,
    )

    def chunk(self, doc: ProcessedDocument, blocks: list[TextBlock]) -> list[Chunk]:
        if not blocks:
            # fallback: same as fixed on plaintext
            return FixedSizeChunker(self.config).chunk(doc, blocks)

        sections = group_by_section(blocks)
        chunks = []
        idx = 0
        for section_heading, section_blocks in sections.items():
            section_text = build_section_text(section_blocks)
            pieces = self._split_recursively(section_text)
            page = pick_page(section_blocks)
            for piece in pieces:
                chunks.append(
                    self._make_chunk(doc, piece, idx, section_heading=section_heading, page_number=page)
                )
                idx += 1
        return chunks