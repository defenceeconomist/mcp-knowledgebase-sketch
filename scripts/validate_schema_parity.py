#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import redis
from qdrant_client import QdrantClient


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _json_list(raw: Any) -> List[Any]:
    text = _decode(raw)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _doc_meta(client: redis.Redis, prefix: str, doc_id: str) -> Dict[str, str]:
    raw = client.hgetall(f"{prefix}:pdf:{doc_id}:meta") or {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        dk = _decode(k)
        dv = _decode(v)
        if dk is None or dv is None:
            continue
        out[str(dk)] = str(dv)
    return out


def _v1_chunk_count(client: redis.Redis, prefix: str, doc_id: str) -> int:
    meta = _doc_meta(client, prefix, doc_id)
    chunks_key = meta.get("chunks_key") or f"{prefix}:pdf:{doc_id}:chunks"
    return len(_json_list(client.get(chunks_key)))


def _v2_chunk_count(client: redis.Redis, prefix: str, doc_id: str) -> int:
    key = f"{prefix}:v2:doc:{doc_id}:chunk_hashes"
    return int(client.zcard(key))


def _all_collections(client: QdrantClient, preferred: str | None, all_collections: bool) -> List[str]:
    if all_collections:
        names = sorted([entry.name for entry in client.get_collections().collections])
        if preferred:
            return sorted(set(names + [preferred]))
        return names
    if preferred:
        return [preferred]
    return [os.getenv("QDRANT_COLLECTION", "pdf_chunks")]


def _qdrant_v2_field_mismatches(
    client: QdrantClient,
    collections: List[str],
    sample_size: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    required = ["doc_hash", "partition_hash", "chunk_hash", "source_id"]
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    per_collection_limit = max(1, sample_size // max(1, len(collections)))
    for collection in collections:
        if not client.collection_exists(collection):
            continue
        offset = None
        taken = 0
        while taken < per_collection_limit:
            points, offset = client.scroll(
                collection_name=collection,
                limit=min(256, per_collection_limit - taken),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                checked += 1
                taken += 1
                payload = point.payload or {}
                missing = [field for field in required if not payload.get(field)]
                if missing:
                    mismatches.append(
                        {
                            "collection": collection,
                            "id": str(point.id),
                            "missing_fields": missing,
                        }
                    )
            if offset is None:
                break
    return checked, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v1/v2 Redis and Qdrant schema parity.")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--redis-prefix", default=os.getenv("REDIS_PREFIX", "unstructured"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "pdf_chunks"))
    parser.add_argument("--all-collections", action="store_true")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    redis_client = redis.from_url(args.redis_url)
    qdrant_client = QdrantClient(url=args.qdrant_url)

    v1_doc_ids = sorted(
        [str(_decode(entry)) for entry in (redis_client.smembers(f"{args.redis_prefix}:pdf:hashes") or set()) if _decode(entry)]
    )
    v2_doc_ids = sorted(
        [str(_decode(entry)) for entry in (redis_client.smembers(f"{args.redis_prefix}:v2:doc_hashes") or set()) if _decode(entry)]
    )
    v1_set = set(v1_doc_ids)
    v2_set = set(v2_doc_ids)

    random.seed(args.seed)
    sample_pool = sorted(v1_set.union(v2_set))
    random.shuffle(sample_pool)
    sample_ids = sample_pool[: max(1, args.sample_size)]

    chunk_mismatches: List[Dict[str, Any]] = []
    for doc_id in sample_ids:
        v1_count = _v1_chunk_count(redis_client, args.redis_prefix, doc_id) if doc_id in v1_set else 0
        v2_count = _v2_chunk_count(redis_client, args.redis_prefix, doc_id) if doc_id in v2_set else 0
        if v1_count != v2_count:
            chunk_mismatches.append(
                {
                    "doc_id": doc_id,
                    "v1_chunks": v1_count,
                    "v2_chunks": v2_count,
                }
            )

    collections = _all_collections(qdrant_client, args.collection, args.all_collections)
    qdrant_checked, qdrant_mismatches = _qdrant_v2_field_mismatches(
        qdrant_client,
        collections=collections,
        sample_size=max(1, args.sample_size),
    )

    summary = {
        "redis": {
            "v1_doc_count": len(v1_set),
            "v2_doc_count": len(v2_set),
            "v1_minus_v2": sorted(v1_set - v2_set)[:50],
            "v2_minus_v1": sorted(v2_set - v1_set)[:50],
            "sample_size": len(sample_ids),
            "chunk_mismatch_count": len(chunk_mismatches),
            "chunk_mismatches": chunk_mismatches[:200],
        },
        "qdrant": {
            "collections": collections,
            "checked_points": qdrant_checked,
            "v2_field_mismatch_count": len(qdrant_mismatches),
            "v2_field_mismatches": qdrant_mismatches[:200],
        },
    }
    print(json.dumps(summary, indent=2))

    failed = (
        len(v1_set) != len(v2_set)
        or bool(v1_set - v2_set)
        or bool(v2_set - v1_set)
        or bool(chunk_mismatches)
        or bool(qdrant_mismatches)
    )
    if failed and args.fail_on_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
