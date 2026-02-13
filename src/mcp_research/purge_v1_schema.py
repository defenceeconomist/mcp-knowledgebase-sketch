from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import redis
from qdrant_client import QdrantClient, models


V1_QDRANT_FIELDS = ("document_id", "source", "pages")


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _batched(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
    out: List[Sequence[Any]] = []
    for idx in range(0, len(items), size):
        out.append(items[idx : idx + size])
    return out


def _target_collections(client: QdrantClient, preferred: str | None, all_collections: bool) -> List[str]:
    if all_collections:
        names = sorted([entry.name for entry in client.get_collections().collections])
        if preferred:
            return sorted(set(names + [preferred]))
        return names
    if preferred:
        return [preferred]
    return [os.getenv("QDRANT_COLLECTION", "pdf_chunks")]


def _count_qdrant_v1_fields(client: QdrantClient, collection_name: str, fields: Sequence[str]) -> Dict[str, int]:
    total = 0
    with_v1 = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            total += 1
            payload = point.payload or {}
            if any(field in payload for field in fields):
                with_v1 += 1
        if offset is None:
            break
    return {"total_points": total, "points_with_v1_fields": with_v1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge Redis/Qdrant v1 schema artifacts.")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--redis-prefix", default=os.getenv("REDIS_PREFIX", "unstructured"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "pdf_chunks"))
    parser.add_argument("--all-collections", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Apply deletions. Default is dry-run.")
    parser.add_argument("--redis-batch-size", type=int, default=1000)
    args = parser.parse_args()

    redis_client = redis.from_url(args.redis_url)
    qdrant_client = QdrantClient(url=args.qdrant_url)

    redis_v1_keys = [
        _decode(key)
        for key in redis_client.scan_iter(match=f"{args.redis_prefix}:pdf:*", count=2000)
    ]

    collections = _target_collections(qdrant_client, args.collection, args.all_collections)
    qdrant_before: Dict[str, Dict[str, int]] = {}
    qdrant_after: Dict[str, Dict[str, int]] = {}
    qdrant_errors: List[str] = []

    for collection_name in collections:
        if not qdrant_client.collection_exists(collection_name):
            qdrant_errors.append(f"missing_collection:{collection_name}")
            continue
        qdrant_before[collection_name] = _count_qdrant_v1_fields(
            qdrant_client,
            collection_name,
            V1_QDRANT_FIELDS,
        )

    redis_deleted = 0
    if args.apply and redis_v1_keys:
        for batch in _batched(redis_v1_keys, max(1, args.redis_batch_size)):
            redis_deleted += int(redis_client.delete(*batch))

    qdrant_payload_updates = 0
    if args.apply:
        for collection_name in qdrant_before:
            if qdrant_before[collection_name].get("points_with_v1_fields", 0) == 0:
                qdrant_after[collection_name] = qdrant_before[collection_name]
                continue
            qdrant_client.delete_payload(
                collection_name=collection_name,
                keys=list(V1_QDRANT_FIELDS),
                points=models.Filter(must=[]),
                wait=True,
            )
            qdrant_payload_updates += 1
            qdrant_after[collection_name] = _count_qdrant_v1_fields(
                qdrant_client,
                collection_name,
                V1_QDRANT_FIELDS,
            )
    else:
        qdrant_after = qdrant_before

    summary = {
        "apply": bool(args.apply),
        "redis": {
            "v1_key_pattern": f"{args.redis_prefix}:pdf:*",
            "v1_key_count": len(redis_v1_keys),
            "deleted": redis_deleted,
        },
        "qdrant": {
            "fields_removed": list(V1_QDRANT_FIELDS),
            "collections": collections,
            "before": qdrant_before,
            "after": qdrant_after,
            "payload_updates": qdrant_payload_updates,
            "errors": qdrant_errors,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
