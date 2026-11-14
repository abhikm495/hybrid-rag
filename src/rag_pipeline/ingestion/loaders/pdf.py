from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from rag_pipeline.ingestion.loaders.base import BaseLoader
from rag_pipeline.ingestion.models import (
    DocumentFormat,
    DocumentMetadata,
    LoaderResult,
    TextBlock,
)
from rag_pipeline.ingestion.normalize import fix_pdf_hyphenation  # you add this

PARA_SPLIT = re.compile(r"\n\s*\n+")

def _page_to_paragraphs(page_text: str) -> list[str]:
    page_text = page_text.strip()
    if not page_text:
        return []
    parts = PARA_SPLIT.split(page_text)
    return [p.strip() for p in parts if p.strip()]

def _extract_pages(path: Path) -> list[tuple[int, str]]:
    """Returns list of (page_number_1based, text)."""
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = fix_pdf_hyphenation(raw)
            pages.append((i, cleaned))
    return pages

def _pages_to_blocks(pages: list[tuple[int, str]]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    heading_stack: list[str] = []  # optional until Step G

    def current_section() -> str:
        return " > ".join(heading_stack) if heading_stack else "Root"

    for page_num, page_text in pages:
        section = current_section() if heading_stack else f"Page {page_num}"

        for para in _page_to_paragraphs(page_text):
            blocks.append(
                TextBlock(
                    kind="paragraph",
                    text=para,
                    section_heading=section,
                    heading_level=None,
                    page_number=page_num,
                )
            )
    return blocks

class PdfLoader(BaseLoader):
    format = DocumentFormat.PDF

    def load(self, path: Path) -> LoaderResult:
        pages = _extract_pages(path)
        page_count = len(pages)
        blocks = _pages_to_blocks(pages)
        plaintext = "\n\n".join(b.text for b in blocks)

        doc_title = self._guess_title(blocks) or path.stem
        metadata = DocumentMetadata(
            source_file=str(path),
            section_heading=doc_title,
            page_number=None,  # document-level metadata stays null
            format=self.format,
            page_count=page_count,
        )
        return LoaderResult(plaintext=plaintext, metadata=metadata, blocks=blocks)

    def load_text(self, path: Path) -> str:
        return self.load(path).plaintext

    @staticmethod
    def _guess_title(blocks: list[TextBlock]) -> str | None:
        if not blocks:
            return None
        first = blocks[0].text.strip()
        if len(first) < 120 and "\n" not in first:
            return first
        return None

def _heading_lines_from_page(page: pdfplumber.page.Page) -> list[tuple[str, int]]:
    chars = page.chars
    if not chars:
        return []
    sizes = [c["size"] for c in chars if c.get("size")]
    if not sizes:
        return []
    median = sorted(sizes)[len(sizes) // 2]
    threshold = median * 1.15
    # Group chars into lines (pdfplumber: page.extract_words() or cluster by 'top')
    # Lines with avg size >= threshold → heading level 2
    ...