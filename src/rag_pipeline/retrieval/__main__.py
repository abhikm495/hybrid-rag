import sys

from rag_pipeline.ingestion.storage import project_data_dir
from rag_pipeline.retrieval.dense import DenseRetriever


def main() -> None:
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    data = project_data_dir()
    retriever = DenseRetriever(chroma_persist_dir=data / "index" / "chroma")

    for hit in retriever.retrieve(question, k=10):
        print(f"[{hit.rank}] score={hit.score:.4f} id={hit.chunk_id}")
        src = hit.metadata.get("source_file", "?")
        print(f"     source={src}")
        print(f"     {hit.text[:200]!r}...")


if __name__ == "__main__":
    main()