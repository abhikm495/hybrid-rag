from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One chunk returned by dense (or later hybrid) search."""

    chunk_id: str
    text: str
    score: float = Field(
        description=(
            "Retrieval score; higher is more relevant. "
            "Dense uses cosine similarity (~0–1); sparse uses BM25 (unbounded)."
        )
    )
    metadata: dict[str, str | int | float | bool]
    rank: int = Field(description="1 = best match in this result list")