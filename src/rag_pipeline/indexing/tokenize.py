from __future__ import annotations

import re

# Words: letters and digits only, lowercased (v1 — no stemming).
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())