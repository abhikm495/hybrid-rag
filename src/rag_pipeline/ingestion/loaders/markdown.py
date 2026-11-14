from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from re import Match

from rag_pipeline.ingestion.loaders.base import BaseLoader, decode_text_bytes
from rag_pipeline.ingestion.models import (
    DocumentFormat,
    DocumentMetadata,
    LoaderResult,
    TextBlock,
)

ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE = re.compile(r"^(\s*)(```|~~~)")


@dataclass
class _ParseState:
    heading_stack: list[str] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)
    para_lines: list[str] = field(default_factory=list)
    in_fence: bool = False
    fence_lines: list[str] = field(default_factory=list)

    def current_section(self) -> str:
        return " > ".join(self.heading_stack) if self.heading_stack else "Root"


class MarkdownLoader(BaseLoader):
    format = DocumentFormat.MARKDOWN

    def load(self, path: Path) -> LoaderResult:
        raw = path.read_bytes()
        source = decode_text_bytes(raw)
        blocks = self._parse_blocks(source)
        plaintext = self._blocks_to_plaintext(blocks)
        doc_section = self._document_title(blocks) or "Root"
        metadata = DocumentMetadata(
            source_file=str(path),
            section_heading=doc_section,
            page_number=None,
            format=self.format,
        )
        return LoaderResult(
            plaintext=plaintext,
            metadata=metadata,
            blocks=blocks,
        )

    def load_text(self, path: Path) -> str:
        return self.load(path).plaintext

    def _document_title(self, blocks: list[TextBlock]) -> str | None:
        for b in blocks:
            if b.kind == "heading" and b.heading_level == 1:
                return b.text
        return None

    def _parse_blocks(self, source: str) -> list[TextBlock]:
        state = _ParseState()
        for line in source.splitlines():
            if self._handle_fence_line(line, state):
                continue
            if state.in_fence:
                state.fence_lines.append(line)
                continue
            if self._handle_heading_line(line, state):
                continue
            if not line.strip():
                self._flush_paragraph(state)
                continue
            state.para_lines.append(line)

        self._flush_paragraph(state)
        if state.in_fence:
            self._flush_code(state)
        return state.blocks

    def _handle_fence_line(self, line: str, state: _ParseState) -> bool:
        if not FENCE.match(line):
            return False
        if state.in_fence:
            self._flush_code(state)
            state.in_fence = False
        else:
            self._flush_paragraph(state)
            state.in_fence = True
        return True

    def _handle_heading_line(self, line: str, state: _ParseState) -> bool:
        match = ATX_HEADING.match(line)
        if not match:
            return False
        self._flush_paragraph(state)
        level = len(match.group(1))
        title = match.group(2).strip()
        self._apply_heading(state.heading_stack, level, title)
        state.blocks.append(
            TextBlock(
                kind="heading",
                text=title,
                section_heading=state.current_section(),
                heading_level=level,
            )
        )
        return True

    @staticmethod
    def _apply_heading(heading_stack: list[str], level: int, title: str) -> None:
        trimmed = heading_stack[: level - 1]
        while len(trimmed) < level:
            trimmed.append("")
        trimmed[level - 1] = title
        heading_stack[:] = trimmed[:level]

    def _flush_paragraph(self, state: _ParseState) -> None:
        if not state.para_lines:
            return
        text = "\n".join(state.para_lines).strip()
        if text:
            state.blocks.append(
                TextBlock(
                    kind="paragraph",
                    text=text,
                    section_heading=state.current_section(),
                )
            )
        state.para_lines.clear()

    def _flush_code(self, state: _ParseState) -> None:
        if not state.fence_lines:
            return
        text = "\n".join(state.fence_lines).strip("\n")
        state.blocks.append(
            TextBlock(
                kind="code",
                text=text,
                section_heading=state.current_section(),
            )
        )
        state.fence_lines.clear()

    @staticmethod
    def _blocks_to_plaintext(blocks: list[TextBlock]) -> str:
        return "\n\n".join(b.text for b in blocks)