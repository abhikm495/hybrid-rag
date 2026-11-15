import argparse

from rag_pipeline.ingestion.storage import project_data_dir
from rag_pipeline.retrieval.dense import DenseRetriever
from rag_pipeline.retrieval.sparse import SparseRetriever


def _print_hits(hits) -> None:
    for hit in hits:
        print(f"[{hit.rank}] score={hit.score:.4f} id={hit.chunk_id}")
        src = hit.metadata.get("source_file", "?")
        print(f"     source={src}")
        preview = hit.text[:200].replace("\n", " ")
        print(f"     {preview!r}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG retrieval index.")
    parser.add_argument(
        "--mode",
        choices=("dense", "sparse"),
        default="dense",
        help="dense=vector search; sparse=BM25 keyword search (default: dense)",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question text",
    )
    args = parser.parse_args()

    question = (
        " ".join(args.question)
        if args.question
        else "What is this document about?"
    )
    data = project_data_dir()

    if args.mode == "sparse":
        retriever = SparseRetriever(
            bm25_dir=data / "index" / "bm25",
            chroma_persist_dir=data / "index" / "chroma",
        )
    else:
        retriever = DenseRetriever(chroma_persist_dir=data / "index" / "chroma")

    print(f"mode={args.mode} question={question!r}")
    _print_hits(retriever.retrieve(question, k=3))


if __name__ == "__main__":
    main()