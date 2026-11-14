from rag_pipeline.chunking.base import Chunker
from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.chunking.models import ChunkingConfig, ChunkingStrategy
from rag_pipeline.chunking.recursive import RecursiveChunker
from rag_pipeline.chunking.semantic import SemanticChunker


def get_chunker(strategy: ChunkingStrategy, config: ChunkingConfig) -> Chunker:
    if strategy == ChunkingStrategy.FIXED:
        return FixedSizeChunker(config)
    if strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(config)
    if strategy == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(config)
    raise ValueError(strategy)