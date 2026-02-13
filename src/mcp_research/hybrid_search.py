from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from mcp_research.runtime_utils import load_dotenv


def _to_list(x) -> List[float]:
    """Convert embeddings to plain Python lists."""
    # FastEmbed may return numpy arrays; keep it dependency-light.
    return x.tolist() if hasattr(x, "tolist") else list(x)


def ensure_collection(client: QdrantClient, collection_name: str, dense_dim: int) -> None:
    """
    Create a collection with:
      - named dense vector:  "dense"
      - named sparse vector: "sparse" (with IDF modifier enabled)
    """
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # recommended for BM25-like sparse models
            )
        },
    )


def recreate_collection(client: QdrantClient, collection_name: str, dense_dim: int) -> None:
    """Drop and recreate a Qdrant collection for a fresh index."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    ensure_collection(client, collection_name, dense_dim)


def embed_corpus(
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    docs: Iterable[str],
) -> Tuple[List[List[float]], List[models.SparseVector]]:
    """Embed a corpus into dense and sparse vectors."""
    dense_vectors: List[List[float]] = []
    sparse_vectors: List[models.SparseVector] = []

    dense_embs = list(dense_model.embed(list(docs)))
    sparse_embs = list(sparse_model.embed(list(docs)))

    for d, s in zip(dense_embs, sparse_embs):
        dense_vectors.append(_to_list(d))
        sparse_vectors.append(
            models.SparseVector(indices=list(s.indices), values=list(s.values))
        )

    return dense_vectors, sparse_vectors


def upsert_docs(
    client: QdrantClient,
    collection_name: str,
    docs: List[str],
    dense_vectors: List[List[float]],
    sparse_vectors: List[models.SparseVector],
) -> None:
    """Upsert demo documents with precomputed vectors into Qdrant."""
    points = []
    for i, (text, dv, sv) in enumerate(zip(docs, dense_vectors, sparse_vectors)):
        points.append(
            models.PointStruct(
                id=i,
                vector={
                    "dense": dv,
                    "sparse": sv,
                },
                payload={"text": text},
            )
        )

    client.upsert(collection_name=collection_name, points=points)


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    query_text: str,
    top_k: int,
    prefetch_k: int,
):
    """Run a hybrid dense+sparse query with reciprocal rank fusion."""
    q_dense = _to_list(next(iter(dense_model.embed([query_text]))))
    q_sparse_emb = next(iter(sparse_model.embed([query_text])))
    q_sparse = models.SparseVector(
        indices=list(q_sparse_emb.indices),
        values=list(q_sparse_emb.values),
    )

    # Universal Query API + Reciprocal Rank Fusion (RRF)
    # This matches Qdrant’s “prefetch + fusion” hybrid query pattern. :contentReference[oaicite:2]{index=2}
    res = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=q_sparse, using="sparse", limit=prefetch_k),
            models.Prefetch(query=q_dense, using="dense", limit=prefetch_k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return res


def main():
    """CLI entry point for running hybrid search demo queries."""
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(description="Hybrid (dense+sparse) search with Qdrant (local).")
    parser.add_argument("query", help="Query text to search for")
    parser.add_argument(
        "--url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "hybrid_demo"),
        help="Collection name",
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
    parser.add_argument("--top-k", type=int, default=5, help="Results to return")
    parser.add_argument("--prefetch-k", type=int, default=40, help="Candidates per retriever before fusion")
    parser.add_argument("--recreate", action="store_true", help="Drop + recreate collection and reindex demo docs")
    args = parser.parse_args()

    client = QdrantClient(url=args.url)

    # Initialise embedding models (downloads on first run)
    dense_model = TextEmbedding(model_name=args.dense_model)
    sparse_model = SparseTextEmbedding(model_name=args.sparse_model)

    # Use one embedding to infer dimension
    dense_dim = len(_to_list(next(iter(dense_model.embed(["dimension probe"])))))

    if args.recreate:
        recreate_collection(client, args.collection, dense_dim)
    else:
        ensure_collection(client, args.collection, dense_dim)

    # Demo corpus (replace with your own loader)
    docs = [
        "Qdrant supports hybrid search by combining dense and sparse vectors.",
        "Sparse vectors are useful for keyword matching (BM25 / SPLADE-style).",
        "Dense embeddings are useful for semantic similarity.",
        "Reciprocal Rank Fusion (RRF) combines multiple ranked lists.",
        "Docker Compose makes it easy to run Qdrant locally.",
    ]

    # If collection is empty, index the demo docs once
    info = client.get_collection(args.collection)
    if (info.points_count or 0) == 0:
        dv, sv = embed_corpus(dense_model, sparse_model, docs)
        upsert_docs(client, args.collection, docs, dv, sv)

    res = hybrid_search(
        client=client,
        collection_name=args.collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        query_text=args.query,
        top_k=args.top_k,
        prefetch_k=args.prefetch_k,
    )

    # QueryResponse contains the ranked points; each is a ScoredPoint. :contentReference[oaicite:3]{index=3}
    print("\nTop results:\n")
    for rank, p in enumerate(res.points, start=1):
        text = (p.payload or {}).get("text", "")
        print(f"{rank:>2}. score={p.score:.4f}  id={p.id}  text={text}")


if __name__ == "__main__":
    main()
