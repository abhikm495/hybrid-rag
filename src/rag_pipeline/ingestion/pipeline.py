from __future__ import annotations

from pathlib import Path

from rag_pipeline.ingestion.models import ProcessedDocument
from rag_pipeline.ingestion.normalize import normalize_plaintext
from rag_pipeline.ingestion.registry import LOADERS
from rag_pipeline.ingestion.storage import (
    compute_document_id,
    project_data_dir,
    save_processed,
    save_raw,
)


def ingest_file(path: Path, data_dir: Path | None = None) -> ProcessedDocument:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"No loader for {ext}")

    loader = LOADERS[ext]
    file_bytes = path.read_bytes()
    document_id = compute_document_id(file_bytes)

    data = data_dir or project_data_dir()

    save_raw(
        data,
        document_id,
        path.name,
        file_bytes,
        source_path=str(path),
        file_format=loader.format.value,
    )

    result = loader.load(path)
    plaintext = normalize_plaintext(result.plaintext)
    metadata = result.metadata

    doc = ProcessedDocument(
        document_id=document_id,
        plaintext=plaintext,
        metadata=metadata,
    )
    save_processed(data, doc, blocks=result.blocks or None)
    return doc