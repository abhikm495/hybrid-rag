import sys
from pathlib import Path

from rag_pipeline.ingestion.pipeline import ingest_file


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m rag_pipeline.ingestion path/to/file.txt")
        raise SystemExit(1)
    doc = ingest_file(Path(sys.argv[1]))
    print(f"Ingested document_id={doc.document_id}")
    print(f"Chars: {len(doc.plaintext)}")


if __name__ == "__main__":
    main()