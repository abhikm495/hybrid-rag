from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline.ingestion.models import ProcessedDocument


def project_data_dir(base: Path | None = None) -> Path:
    """Default data/ folder at repo root (parent of src/)."""
    if base is not None:
        return base
    # storage.py → ingestion → rag_pipeline → src → repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data"


def compute_document_id(file_bytes: bytes) -> str:
    """Stable ID from content. Same file content → same id."""
    digest = hashlib.sha256(file_bytes).hexdigest()
    return digest[:16]


def raw_dir(data_dir: Path, document_id: str) -> Path:
    return data_dir / "raw" / document_id


def processed_dir(data_dir: Path, document_id: str) -> Path:
    return data_dir / "processed" / document_id


def save_raw(
    data_dir: Path,
    document_id: str,
    original_filename: str,
    file_bytes: bytes,
    source_path: str,
    file_format: str,
) -> Path:
    """Copy bytes to data/raw/{id}/original.* and write manifest.json."""
    dest = raw_dir(data_dir, document_id)
    dest.mkdir(parents=True, exist_ok=True)

    original_path = dest / f"original{Path(original_filename).suffix}"
    original_path.write_bytes(file_bytes)

    manifest = {
        "document_id": document_id,
        "source_path": source_path,
        "format": file_format,
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "original_filename": original_path.name,
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return original_path


def save_processed(
    data_dir: Path,
    doc: ProcessedDocument,
    blocks: list | None = None,
) -> Path:
    dest = processed_dir(data_dir, doc.document_id)
    dest.mkdir(parents=True, exist_ok=True)

    (dest / "plaintext.txt").write_text(doc.plaintext, encoding="utf-8")

    meta = {
        "document_id": doc.document_id,
        "processing_version": doc.processing_version,
        "metadata": doc.metadata.model_dump(mode="json"),
    }
    (dest / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if blocks:
        payload = [b.model_dump(mode="json") for b in blocks]
        (dest / "blocks.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    return dest


def load_processed_plaintext(data_dir: Path, document_id: str) -> str:
    """Used later for re-index without re-ingesting."""
    path = processed_dir(data_dir, document_id) / "plaintext.txt"
    return path.read_text(encoding="utf-8")