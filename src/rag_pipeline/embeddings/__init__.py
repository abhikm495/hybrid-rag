from __future__ import annotations

import hashlib
import math
import os

# Set RAG_MOCK_EMBEDDINGS=1 to skip the API while testing plumbing.
_MOCK_DIM = 256


def _mock_embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals: list[float] = []
    while len(vals) < _MOCK_DIM:
        for b in digest:
            vals.append((b / 255.0) * 2.0 - 1.0)
            if len(vals) >= _MOCK_DIM:
                break
        digest = hashlib.sha256(digest).digest()
    norm = math.sqrt(sum(x * x for x in vals)) or 1.0
    return [x / norm for x in vals]


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    if not texts:
        return []

    if os.environ.get("RAG_MOCK_EMBEDDINGS") == "1":
        return [_mock_embedding(t) for t in texts]

    from openai import OpenAI

    client = OpenAI()
    out: list[list[float]] = []
    batch_size = 64

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        out.extend(item.embedding for item in response.data)

    return out