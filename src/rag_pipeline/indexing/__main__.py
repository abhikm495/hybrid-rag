from pathlib import Path

from rag_pipeline.chunking.factory import get_chunker
from rag_pipeline.chunking.io import load_processed
from rag_pipeline.chunking.models import ChunkingConfig, ChunkingStrategy
from rag_pipeline.ingestion.storage import project_data_dir
from rag_pipeline.indexing.pipeline import index_chunks

def main():
    document_id = "9d41307cad8d3bd4"  # later: argparse
    strategy = ChunkingStrategy.FIXED

    data = project_data_dir()
    doc, blocks = load_processed(document_id, data)
    chunker = get_chunker(strategy, ChunkingConfig())
    chunks = chunker.chunk(doc, blocks)

    stats = index_chunks(chunks, data_dir=data)
    print(stats)

if __name__ == "__main__":
    main()