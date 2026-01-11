import argparse
import os
import sys

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import hybrid_search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query an existing Qdrant collection with hybrid search.",
    )
    parser.add_argument("query", help="Query text")
    parser.add_argument(
        "collection",
        nargs="?",
        default=os.getenv("QDRANT_COLLECTION", "hybrid_demo"),
        help="Collection name (default: QDRANT_COLLECTION or hybrid_demo)",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Results to return")
    parser.add_argument(
        "--prefetch-k",
        type=int,
        default=40,
        help="Candidates per retriever before fusion",
    )
    parser.add_argument(
        "--dense-model",
        default=os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5"),
        help="FastEmbed dense model name",
    )
    parser.add_argument(
        "--sparse-model",
        default=os.getenv("SPARSE_MODEL", "Qdrant/bm25"),
        help="FastEmbed sparse model name",
    )
    args = parser.parse_args()

    client = QdrantClient(url=args.url)
    dense_model = TextEmbedding(model_name=args.dense_model)
    sparse_model = SparseTextEmbedding(model_name=args.sparse_model)

    result = hybrid_search.hybrid_search(
        client=client,
        collection_name=args.collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        query_text=args.query,
        top_k=args.top_k,
        prefetch_k=args.prefetch_k,
    )

    print("Top results:\n")
    for rank, point in enumerate(result.points, start=1):
        text = (point.payload or {}).get("text", "")
        print(f"{rank:>2}. score={point.score:.4f} id={point.id} text={text}")


if __name__ == "__main__":
    main()
