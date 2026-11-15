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
        self._texts: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._text_by_id: dict[str, str] = {}

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)

    def rebuild(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._texts = [c.text for c in chunks]
        self._text_by_id = dict(zip(self._chunk_ids, self._texts, strict=True))
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

        self._texts = corpus_texts
        self._text_by_id = dict(zip(self._chunk_ids, self._texts, strict=True))
        self._tokenized_corpus = [tokenize(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def merge_and_save(self, dir_path: Path, new_chunks: list[Chunk]) -> None:
        """Load existing BM25 index if present, append new chunk_ids, rebuild, save."""
        existing = self._load_chunks_for_rebuild(dir_path)
        existing_ids = {c.chunk_id for c in existing}
        merged = existing + [c for c in new_chunks if c.chunk_id not in existing_ids]
        if not merged:
            return
        self.rebuild(merged)
        self.save(dir_path, merged)

    def _load_chunks_for_rebuild(self, dir_path: Path) -> list[Chunk]:
        ids_path = dir_path / "chunk_ids.json"
        corpus_path = dir_path / "corpus.jsonl"
        if not ids_path.is_file() or not corpus_path.is_file():
            return []

        from rag_pipeline.chunking.models import ChunkMetadata, ChunkingStrategy

        chunks: list[Chunk] = []
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row["text"]
                chunks.append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        text=text,
                        metadata=ChunkMetadata(
                            source_file="",
                            section_heading="",
                            chunk_index=0,
                            chunking_strategy=ChunkingStrategy.FIXED,
                            char_count=len(text),
                            document_id="",
                        ),
                    )
                )
        return chunks

    def text_by_id(self, chunk_id: str) -> str:
        try:
            return self._text_by_id[chunk_id]
        except KeyError as exc:
            raise KeyError(f"unknown chunk_id: {chunk_id}") from exc

    def top_k(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("Index not built; call rebuild() or load() first")
        if k <= 0:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
        return [(self._chunk_ids[i], float(score)) for i, score in ranked]