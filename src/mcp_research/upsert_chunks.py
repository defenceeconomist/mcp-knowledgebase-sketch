import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, List

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _to_list(x) -> List[float]:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


def batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def load_chunk_items(chunks_dir: Path) -> List[dict]:
    items: List[dict] = []
    for chunk_file in sorted(chunks_dir.glob("*.json")):
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            logger.warning("Skipping %s (expected list payload)", chunk_file)
            continue
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            items.append(entry)
    return items


def upsert_items(
    client: QdrantClient,
    collection: str,
    items: List[dict],
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    batch_size: int,
) -> int:
    if not items:
        return 0

    total = 0
    for batch in batched(items, batch_size):
        texts = [item["text"] for item in batch]
        dense_embs = list(dense_model.embed(texts))
        sparse_embs = list(sparse_model.embed(texts))

        points = []
        for item, dense_vec, sparse_vec in zip(batch, dense_embs, sparse_embs):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": _to_list(dense_vec),
                        "sparse": models.SparseVector(
                            indices=list(sparse_vec.indices),
                            values=list(sparse_vec.values),
                        ),
                    },
                    payload={
                        "source": item.get("source"),
                        "chunk_index": item.get("chunk_index"),
                        "pages": item.get("pages") or [],
                        "text": item.get("text") or "",
                    },
                )
            )

        client.upsert(collection_name=collection, points=points)
        total += len(points)

    return total


def main() -> None:
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Upsert chunk JSON files into Qdrant for hybrid search.",
    )
    parser.add_argument(
        "--chunks-dir",
        default=os.getenv("CHUNKS_DIR", "data/chunks"),
        help="Directory containing chunk JSON files",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "pdf_chunks"),
        help="Qdrant collection name",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Upsert batch size",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop + recreate the collection before upserting",
    )
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir).expanduser()
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    client = QdrantClient(url=args.url)
    dense_model = TextEmbedding(model_name=args.dense_model)
    sparse_model = SparseTextEmbedding(model_name=args.sparse_model)

    dense_dim = len(_to_list(next(iter(dense_model.embed(["dimension probe"])))))
    if args.recreate and client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    ensure_collection(client, args.collection, dense_dim)

    items = load_chunk_items(chunks_dir)
    total = upsert_items(
        client=client,
        collection=args.collection,
        items=items,
        dense_model=dense_model,
        sparse_model=sparse_model,
        batch_size=args.batch_size,
    )
    logger.info("Upserted %d chunks into '%s'", total, args.collection)


if __name__ == "__main__":
    main()
