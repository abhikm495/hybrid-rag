from typing import Protocol
from rag_pipeline.ingestion.models import ProcessedDocument, TextBlock
from rag_pipeline.chunking.models import Chunk

class Chunker(Protocol):
    def chunk(
        self,
        doc: ProcessedDocument,
        blocks: list[TextBlock],
    ) -> list[Chunk]:
        ...