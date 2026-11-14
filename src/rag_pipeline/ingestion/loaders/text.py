from __future__ import annotations

from pathlib import Path

from rag_pipeline.ingestion.models import DocumentFormat, DocumentMetadata
from rag_pipeline.ingestion.loaders.base import BaseLoader, decode_text_bytes


class TextLoader(BaseLoader):
    format = DocumentFormat.TEXT

    def load_text(self, path: Path) -> str:
        raw = path.read_bytes()
        return decode_text_bytes(raw)

    def metadata_for(self, path: Path) -> DocumentMetadata:
        return DocumentMetadata(
            source_file=str(path),
            section_heading="Root",
            page_number=None,
            format=self.format,
        )