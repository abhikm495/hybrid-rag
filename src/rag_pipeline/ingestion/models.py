from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel,Field

class DocumentFormat(str,Enum):
    """while file type we ingested. str mixin makes JSON serialization easy."""

    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"

class DocumentMetadata(BaseModel):
    """Labels that travel with text into chunking and citations"""

    source_file : str = Field(...,description="Path or filename humans recognize")
    section_heading : str = Field(default = "Root",description="Best section title for this text (per-block later)")
    page_number : Optional[int] = Field(default = None,description = "1-based page in PDF; null for md/txt/html",)
    format : DocumentFormat
    page_count: Optional[int] = Field(default=None,description="Total pages for PDF; null for other formats",
)

class ProcessedDocument(BaseModel):
    """Result of ingestion: clean text + metadata + stable id."""

    document_id : str
    plaintext: str
    metadata: DocumentMetadata
    processing_version: str = "v1"
     
from typing import Literal, Optional

BlockKind = Literal["heading", "paragraph", "code"]


class TextBlock(BaseModel):
    """One structured piece of a processed document."""

    kind: BlockKind
    text: str
    section_heading: str
    heading_level: Optional[int] = Field(
        default=None,
        description="1-6 for ATX headings; null for paragraph/code",
    )
    page_number: Optional[int] = Field(
        default=None,
        description="1-based PDF page; null for md/txt/html",
    )


class LoaderResult(BaseModel):
    """Everything a loader produces in one pass."""

    plaintext: str
    metadata: DocumentMetadata
    blocks: list[TextBlock] = Field(default_factory=list)