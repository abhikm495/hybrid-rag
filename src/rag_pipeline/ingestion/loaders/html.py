from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import SoupStrainer  # optional, not required in v1

from rag_pipeline.ingestion.loaders.base import BaseLoader, decode_text_bytes
from rag_pipeline.ingestion.models import (
    DocumentFormat,
    DocumentMetadata,
    LoaderResult,
    TextBlock,
)

STRIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}
NOISE_TAGS = {"nav", "footer"}  # tunable: drop "header" if it steals site chrome

BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre")
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

def _remove_tags(root: Tag, names: set[str]) -> None:
    for name in names:
        for tag in root.find_all(name):
            tag.decompose()  # removes node and children from tree

def _content_root(soup: BeautifulSoup) -> Tag:
    main = soup.find("main")
    if main:
        return main
    article = soup.find("article")
    if article:
        return article
    body = soup.find("body")
    if body:
        return body
    return soup  # fallback

def _visible_text(tag: Tag) -> str:
    text = tag.get_text(separator=" ", strip=True)
    return " ".join(text.split())  # collapse internal whitespace

def _apply_heading(stack: list[str], level: int, title: str) -> None:
    while len(stack) < level:
        stack.append("")
    stack[level - 1] = title
    del stack[level:]  # drop deeper levels

def _parse_blocks(root: Tag) -> list[TextBlock]:
    heading_stack: list[str] = []
    blocks: list[TextBlock] = []

    def current_section() -> str:
        return " > ".join(heading_stack) if heading_stack else "Root"

    for tag in root.find_all(BLOCK_TAGS):
        if tag.name in HEADING_TAGS:
            if tag.find_parent("pre"):
                continue
            level = int(tag.name[1])
            title = _visible_text(tag)
            if not title:
                continue
            _apply_heading(heading_stack, level, title)
            blocks.append(
                TextBlock(
                    kind="heading",
                    text=title,
                    section_heading=current_section(),
                    heading_level=level,
                )
            )
        elif tag.name == "pre":
            code = tag.get_text("\n", strip=False).strip("\n")
            if code:
                blocks.append(
                    TextBlock(
                        kind="code",
                        text=code,
                        section_heading=current_section(),
                    )
                )
        elif tag.name in ("p", "li"):
            if tag.find_parent("pre"):
                continue
            text = _visible_text(tag)
            if text:
                blocks.append(
                    TextBlock(
                        kind="paragraph",
                        text=text,
                        section_heading=current_section(),
                    )
                )

    return blocks

class HtmlLoader(BaseLoader):
    format = DocumentFormat.HTML

    def load(self, path: Path) -> LoaderResult:
        raw = path.read_bytes()
        html = decode_text_bytes(raw)
        soup = BeautifulSoup(html, "lxml")

        for tag in soup.find_all(list(STRIP_TAGS | NOISE_TAGS)):
            if tag.name in STRIP_TAGS | NOISE_TAGS:
                tag.decompose()

        root = _content_root(soup)
        blocks = _parse_blocks(root)
        plaintext = "\n\n".join(b.text for b in blocks)

        title_tag = soup.find("title")
        doc_title = (
            _visible_text(title_tag)
            if title_tag and title_tag.get_text(strip=True)
            else None
        )
        if not doc_title:
            doc_title = self._first_h1(blocks) or "Root"

        metadata = DocumentMetadata(
            source_file=str(path),
            section_heading=doc_title,
            page_number=None,
            format=self.format,
        )
        return LoaderResult(plaintext=plaintext, metadata=metadata, blocks=blocks)

    def load_text(self, path: Path) -> str:
        return self.load(path).plaintext

    @staticmethod
    def _first_h1(blocks: list[TextBlock]) -> str | None:
        for b in blocks:
            if b.kind == "heading" and b.heading_level == 1:
                return b.text
        return None

