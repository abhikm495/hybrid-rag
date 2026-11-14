from __future__ import annotations

from pathlib import Path

from rag_pipeline.ingestion.loaders.html import HtmlLoader
from rag_pipeline.ingestion.loaders.markdown import MarkdownLoader
from rag_pipeline.ingestion.loaders.pdf import PdfLoader
from rag_pipeline.ingestion.models import DocumentFormat
from rag_pipeline.ingestion.loaders.text import TextLoader

# We'll add MarkdownLoader, HtmlLoader, PdfLoader later
LOADERS = {
    ".txt": TextLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".html": HtmlLoader(),
    ".htm": HtmlLoader(),
    ".pdf": PdfLoader(),
}


def detect_format(path: Path) -> DocumentFormat:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported extension: {ext}")
    loader = LOADERS[ext]
    return loader.format