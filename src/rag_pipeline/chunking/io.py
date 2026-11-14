from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline.ingestion.models import DocumentMetadata, ProcessedDocument, TextBlock
from rag_pipeline.ingestion.storage import processed_dir, project_data_dir


def load_processed(
    document_id: str,
    data_dir: Path | None = None,
) -> tuple[ProcessedDocument, list[TextBlock]]:
    data = data_dir or project_data_dir()
    folder = processed_dir(data, document_id)

    if not folder.is_dir():
        raise FileNotFoundError(f"No processed folder for document_id={document_id}: {folder}")

    meta_path = folder / "metadata.json"
    plain_path = folder / "plaintext.txt"
    blocks_path = folder / "blocks.json"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.json: {meta_path}")
    if not plain_path.is_file():
        raise FileNotFoundError(f"Missing plaintext.txt: {plain_path}")

  # --- metadata.json ---
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc_metadata = DocumentMetadata.model_validate(raw_meta["metadata"])
    processing_version = raw_meta.get("processing_version", "v1")
    file_document_id = raw_meta.get("document_id", document_id)

    # --- plaintext.txt ---
    plaintext = plain_path.read_text(encoding="utf-8")

    doc = ProcessedDocument(
        document_id=file_document_id,
        plaintext=plaintext,
        metadata=doc_metadata,
        processing_version=processing_version,
    )

    # --- blocks.json (optional) ---
    blocks: list[TextBlock] = []
    if blocks_path.is_file():
        raw_blocks = json.loads(blocks_path.read_text(encoding="utf-8"))
        blocks = [TextBlock.model_validate(b) for b in raw_blocks]

    return doc, blocks