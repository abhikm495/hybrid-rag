from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.indexing.chroma_store import ChromaChunkStore

DEDUP_SIMILARITY_THRESHOLD = 0.95


@dataclass(frozen=True)
class DedupDecision:
    chunk_id: str
    keep: bool
    duplicate_of: str | None = None
    similarity: float | None = None
    source_file: str | None = None


def should_keep(
    similarity: float | None, *, threshold: float = DEDUP_SIMILARITY_THRESHOLD
) -> bool:
    if similarity is None:
        return True
    return similarity <= threshold


def filter_near_duplicates(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    store: ChromaChunkStore,
    *,
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    upsert_kept: bool = True,
) -> tuple[list[Chunk], list[list[float]], list[DedupDecision]]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    kept_chunks: list[Chunk] = []
    kept_embeddings: list[list[float]] = []
    decisions: list[DedupDecision] = []

    for chunk, emb in zip(chunks, embeddings):
        neighbor = store.find_nearest_neighbor(emb)
        sim = neighbor[1] if neighbor else None
        source_file = chunk.metadata.source_file
        dup_id = neighbor[0] if neighbor else None
        is_same_chunk = dup_id == chunk.chunk_id

        if neighbor and not is_same_chunk and not should_keep(sim, threshold=threshold):
            decisions.append(
                DedupDecision(
                    chunk_id=chunk.chunk_id,
                    keep=False,
                    duplicate_of=dup_id,
                    similarity=sim,
                    source_file=source_file,
                )
            )
            continue

        kept_chunks.append(chunk)
        kept_embeddings.append(emb)
        if upsert_kept:
            store.upsert_chunks([chunk], [emb])
        decisions.append(
            DedupDecision(
                chunk_id=chunk.chunk_id,
                keep=True,
                similarity=sim,
                source_file=source_file,
            )
        )

    return kept_chunks, kept_embeddings, decisions


def append_dedup_log(decisions: list[DedupDecision], log_path: Path) -> None:
    skipped = [d for d in decisions if not d.keep]
    if not skipped:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for d in skipped:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")