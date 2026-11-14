from __future__ import annotations

import math

from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.chunking.models import Chunk, ChunkMetadata, ChunkingConfig, ChunkingStrategy
from rag_pipeline.chunking.recursive import RecursiveChunker, build_section_text, pick_page
from rag_pipeline.embeddings import embed_texts
from rag_pipeline.ingestion.models import ProcessedDocument, TextBlock


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_split(
    blocks: list[TextBlock],
    vectors: list[list[float]],
    threshold: float,
) -> list[list[TextBlock]]:
    if not blocks:
        return []
    if len(blocks) != len(vectors):
        raise ValueError("blocks and vectors must have the same length")

    groups: list[list[TextBlock]] = []
    current = [blocks[0]]

    for i in range(1, len(blocks)):
        sim = cosine(vectors[i - 1], vectors[i])
        if sim < threshold:
            groups.append(current)
            current = [blocks[i]]
        else:
            current.append(blocks[i])
    groups.append(current)
    return groups


class SemanticChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._recursive = RecursiveChunker(config)

    def _make_chunk(
        self,
        doc: ProcessedDocument,
        piece: str,
        index: int,
        *,
        section_heading: str,
        page_number: int | None,
    ) -> Chunk:
        meta = ChunkMetadata(
            source_file=doc.metadata.source_file,
            section_heading=section_heading,
            page_number=page_number,
            chunk_index=index,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            char_count=len(piece),
            document_id=doc.document_id,
        )
        return Chunk(
            chunk_id=f"{doc.document_id}:semantic:{index}",
            text=piece,
            metadata=meta,
        )

    def chunk(self, doc: ProcessedDocument, blocks: list[TextBlock]) -> list[Chunk]:
        if not blocks:
            return FixedSizeChunker(self.config).chunk(doc, blocks)

        usable = [b for b in blocks if b.text.strip()]
        if not usable:
            return FixedSizeChunker(self.config).chunk(doc, blocks)

        texts = [b.text for b in usable]
        vectors = embed_texts(texts, self.config.embedding_model)
        groups = semantic_split(usable, vectors, self.config.similarity_threshold)

        chunks: list[Chunk] = []
        idx = 0

        for group in groups:
            group_text = build_section_text(group)
            if not group_text.strip():
                continue

            section_heading = group[0].section_heading or doc.metadata.section_heading or "Root"
            page_number = pick_page(group)

            pieces = self._recursive._split_recursively(group_text)
            for piece in pieces:
                chunks.append(
                    self._make_chunk(
                        doc,
                        piece,
                        idx,
                        section_heading=section_heading,
                        page_number=page_number,
                    )
                )
                idx += 1

        return chunks if chunks else FixedSizeChunker(self.config).chunk(doc, blocks)