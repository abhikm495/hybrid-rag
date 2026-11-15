from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One chunk returned by dense (or later hybrid) search."""

    chunk_id: str
    text: str
    score: float = Field(
        description="Cosine similarity in ~[0, 1]; higher is more relevant."
    )
    metadata: dict[str, str | int | float | bool]
    rank: int = Field(description="1 = best match in this result list")