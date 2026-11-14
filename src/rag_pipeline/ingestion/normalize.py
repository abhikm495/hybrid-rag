from __future__ import annotations

import re
import unicodedata

# "word-\ncontinuation" → "wordcontinuation" (often you want no hyphen: word + continuation)
HYPHEN_LINE_BREAK = re.compile(r"(\w)-\n(\w)")

def fix_pdf_hyphenation(text: str) -> str:
  return HYPHEN_LINE_BREAK.sub(r"\1\2", text)


def normalize_plaintext(text: str) -> str:
    """Make text consistent for chunking and BM25."""
    if not text:
        return ""

    # Unicode standard form (accents, compatibility characters)
    text = unicodedata.normalize("NFC", text)

    # Windows / old Mac line endings → single newline
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim trailing spaces on each line
    lines = [line.rstrip() for line in text.split("\n")]

    # Collapse 3+ blank lines to 2 (one empty line between paragraphs)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()