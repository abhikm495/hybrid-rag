from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.indexing.tokenize import tokenize


class BM25ChunkStore:
    """Sparse keyword index over chunks. chunk_ids[i] aligns with corpus row i."""

    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)

    def rebuild(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._tokenized_corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def save(self, dir_path: Path, chunks: list[Chunk]) -> None:
        ids_from_chunks = [c.chunk_id for c in chunks]
        if ids_from_chunks != self._chunk_ids:
            raise ValueError(
                "chunks do not match the last rebuild(); call rebuild(chunks) first"
            )

        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "chunk_ids.json").write_text(
            json.dumps(self._chunk_ids, indent=2),
            encoding="utf-8",
        )
        with (dir_path / "corpus.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                line = json.dumps(
                    {"chunk_id": c.chunk_id, "text": c.text},
                    ensure_ascii=False,
                )
                f.write(line + "\n")

    def load(self, dir_path: Path) -> None:
        ids_path = dir_path / "chunk_ids.json"
        corpus_path = dir_path / "corpus.jsonl"
        if not ids_path.is_file() or not corpus_path.is_file():
            raise FileNotFoundError(f"BM25 index missing under {dir_path}")

        self._chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        corpus_texts: list[str] = []
        file_ids: list[str] = []

        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                file_ids.append(row["chunk_id"])
                corpus_texts.append(row["text"])

        if file_ids != self._chunk_ids:
            raise ValueError("chunk_ids.json does not match corpus.jsonl")

        self._tokenized_corpus = [tokenize(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def top_k(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("Index not built; call rebuild() or load() first")
        if k <= 0:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
        return [(self._chunk_ids[i], float(score)) for i, score in ranked]