from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"

class ChunkingConfig(BaseModel):
  """Knobs shared or per-strategy."""
  chunk_size: int = 800          # characters, not tokens (v1)
  chunk_overlap: int = 120
  # semantic-only:
  similarity_threshold: float = 0.5  # break when adjacent sentences are LESS similar than this
  embedding_model: str = "text-embedding-3-small"

class ChunkMetadata(BaseModel):
    source_file: str
    section_heading: str
    page_number: Optional[int] = None
    chunk_index: int
    chunking_strategy: ChunkingStrategy
    char_count: int
    document_id: str

class Chunk(BaseModel):
    chunk_id: str   # you define: e.g. f"{document_id}:{strategy}:{chunk_index}"
    text: str
    metadata: ChunkMetadata