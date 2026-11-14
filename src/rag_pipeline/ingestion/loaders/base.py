from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from rag_pipeline.ingestion.models import DocumentFormat, DocumentMetadata, LoaderResult


class BaseLoader(ABC):
    format: DocumentFormat

    def load(self, path: Path) -> LoaderResult:
        """Default path: load_text + metadata_for. Markdown overrides."""
        raw_text = self.load_text(path)
        return LoaderResult(
            plaintext=raw_text,
            metadata=self.metadata_for(path),
            blocks=[],
        )

    def metadata_for(self, path: Path) -> DocumentMetadata:
        return DocumentMetadata(
            source_file=str(path),
            section_heading="Root",
            page_number=None,
            format=self.format,
        )
    

    @abstractmethod
    def load_text(self, path: Path) -> str:
        ...

def decode_text_bytes(raw: bytes) -> str:
    """Try UTF-8, then chardet guess — beginner-friendly encoding handling."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        import chardet

        guess = chardet.detect(raw)
        encoding = guess.get("encoding") or "utf-8"
        return raw.decode(encoding, errors="replace")