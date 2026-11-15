from __future__ import annotations

import hashlib
import math
import os

# Set RAG_MOCK_EMBEDDINGS=1 to skip the API while testing plumbing.
_MOCK_DIM = 256
_LOCAL_MODEL: "SentenceTransformer | None" = None


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

def _get_local_model(name: str):
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_MODEL = SentenceTransformer(name)
    return _LOCAL_MODEL

def embed_texts(texts: list[str], model: str) -> list[str]:
    if not texts:
        return []
    if os.environ.get("RAG_MOCK_EMBEDDINGS") == "1":
        return [_mock_embedding(t) for t in texts]
    backend = os.environ.get("RAG_EMBEDDING_BACKEND", "local").lower()
    if backend == "local":
        local_name = os.environ.get("RAG_LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        st = _get_local_model(local_name)
        vecs = st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]
  # backend == "openai"

    from openai import OpenAI

    client = OpenAI()
    out: list[list[float]] = []
    batch_size = 64

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        out.extend(item.embedding for item in response.data)

    return out