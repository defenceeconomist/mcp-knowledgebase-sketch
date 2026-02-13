import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, List

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from mcp_research.ingest_unstructured import record_collection_mapping
from mcp_research.schema_v2 import (
    chunk_hash,
    partition_hash,
    qdrant_payload_schema_mode,
    qdrant_point_id_mode,
    should_write_v1,
    should_write_v2,
)

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_dotenv(path: Path) -> None:
    """Load a .env file into the process environment if present."""
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
    """Convert embeddings to plain Python lists."""
    return x.tolist() if hasattr(x, "tolist") else list(x)


def ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """Ensure a Qdrant collection exists with dense+sparse vector config."""
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
    """Yield items in fixed-size batches."""
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def _get_redis_client(redis_url: str):
    """Return a Redis client for upsert bookkeeping."""
    if not redis_url:
        return None
    if redis is None:
        raise RuntimeError("redis package is required for Redis upserts")
    return redis.from_url(redis_url)


def _decode_redis_value(value):
    """Decode Redis bytes payloads into strings when needed."""
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def load_chunk_items(chunks_dir: Path) -> List[dict]:
    """Load chunk payloads from JSON files on disk."""
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


def load_chunk_items_from_redis(
    redis_client,
    prefix: str,
    doc_ids: List[str] | None,
) -> List[dict]:
    """Load chunk payloads from Redis for the specified document ids."""
    items: List[dict] = []
    if doc_ids:
        ids = doc_ids
    else:
        raw_ids = redis_client.smembers(f"{prefix}:pdf:hashes")
        ids = [_decode_redis_value(val) for val in raw_ids]

    for doc_id in ids:
        if not doc_id:
            continue
        chunks_key = f"{prefix}:pdf:{doc_id}:chunks"
        raw = redis_client.get(chunks_key)
        raw = _decode_redis_value(raw)
        if not raw:
            logger.warning("Missing Redis key %s", chunks_key)
            continue
        payload = json.loads(raw)
        if not isinstance(payload, list):
            logger.warning("Skipping %s (expected list payload)", chunks_key)
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
    """Upsert chunk items into Qdrant and return the number written."""
    if not items:
        return 0

    total = 0
    for batch in batched(items, batch_size):
        texts = [item["text"] for item in batch]
        dense_embs = list(dense_model.embed(texts))
        sparse_embs = list(sparse_model.embed(texts))

        points = []
        for item, dense_vec, sparse_vec in zip(batch, dense_embs, sparse_embs):
            payload_mode = qdrant_payload_schema_mode()
            point_mode = qdrant_point_id_mode()
            doc_hash = str(item.get("doc_hash") or item.get("document_id") or "")
            p_hash = str(item.get("partition_hash") or partition_hash(doc_hash, item))
            c_hash = str(item.get("chunk_hash") or chunk_hash(doc_hash, item))

            payload = {}
            if should_write_v1(payload_mode):
                payload.update(
                    {
                        "document_id": item.get("document_id"),
                        "source": item.get("source"),
                        "chunk_index": item.get("chunk_index"),
                        "pages": item.get("pages") or [],
                        "text": item.get("text") or "",
                    }
                )
                for key in (
                    "source_ref",
                    "bucket",
                    "key",
                    "version_id",
                    "page_start",
                    "page_end",
                ):
                    if key in item and item.get(key) is not None:
                        payload[key] = item.get(key)

            if should_write_v2(payload_mode):
                payload.update(
                    {
                        "doc_hash": doc_hash,
                        "partition_hash": p_hash,
                        "chunk_hash": c_hash,
                    }
                )
                source_id = item.get("source_id")
                if source_id:
                    payload["source_id"] = source_id
                if not payload.get("text"):
                    payload["text"] = item.get("text") or ""
                if "chunk_index" not in payload:
                    payload["chunk_index"] = item.get("chunk_index")
                if "page_start" not in payload and item.get("page_start") is not None:
                    payload["page_start"] = item.get("page_start")
                if "page_end" not in payload and item.get("page_end") is not None:
                    payload["page_end"] = item.get("page_end")

            point_id = str(uuid.uuid4())
            if point_mode == "deterministic":
                base = f"{collection}|{doc_hash}|{c_hash}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, base))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": _to_list(dense_vec),
                        "sparse": models.SparseVector(
                            indices=list(sparse_vec.indices),
                            values=list(sparse_vec.values),
                        ),
                    },
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection, points=points)
        total += len(points)

    return total


def main() -> None:
    """CLI entry point to upsert chunks into Qdrant."""
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Upsert chunk payloads into Qdrant for hybrid search.",
    )
    parser.add_argument(
        "--source",
        choices=["disk", "redis"],
        default=os.getenv("CHUNKS_SOURCE", "disk"),
        help="Load chunks from disk or Redis",
    )
    parser.add_argument(
        "--chunks-dir",
        default=os.getenv("CHUNKS_DIR", "data/chunks"),
        help="Directory containing chunk JSON files",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", ""),
        help="Redis URL (required for --source redis)",
    )
    parser.add_argument(
        "--redis-prefix",
        default=os.getenv("REDIS_PREFIX", "unstructured"),
        help="Redis key prefix for chunk payloads",
    )
    parser.add_argument(
        "--redis-doc-id",
        action="append",
        default=[],
        help="Only upsert specific document_id values (repeatable)",
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

    client = QdrantClient(url=args.url)
    dense_model = TextEmbedding(model_name=args.dense_model)
    sparse_model = SparseTextEmbedding(model_name=args.sparse_model)

    dense_dim = len(_to_list(next(iter(dense_model.embed(["dimension probe"])))))
    if args.recreate and client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    ensure_collection(client, args.collection, dense_dim)

    redis_client_for_load = None
    if args.source == "disk":
        chunks_dir = Path(args.chunks_dir).expanduser()
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
        items = load_chunk_items(chunks_dir)
    else:
        redis_client_for_load = _get_redis_client(args.redis_url)
        if redis_client_for_load is None:
            raise RuntimeError("REDIS_URL is required when --source redis")
        items = load_chunk_items_from_redis(
            redis_client=redis_client_for_load,
            prefix=args.redis_prefix,
            doc_ids=args.redis_doc_id or None,
        )
    redis_client = redis_client_for_load
    if redis_client is None and args.redis_url:
        redis_client = _get_redis_client(args.redis_url)
    total = upsert_items(
        client=client,
        collection=args.collection,
        items=items,
        dense_model=dense_model,
        sparse_model=sparse_model,
        batch_size=args.batch_size,
    )
    if redis_client:
        doc_ids = {item.get("document_id") for item in items if item.get("document_id")}
        for doc_id in doc_ids:
            record_collection_mapping(
                redis_client=redis_client,
                doc_id=doc_id,
                collection=args.collection,
                prefix=args.redis_prefix,
            )
    logger.info("Upserted %d chunks into '%s'", total, args.collection)


if __name__ == "__main__":
    main()
