#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import redis

from mcp_research.schema_v2 import SourceDescriptor, split_source_path, write_v2_document_payloads


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _json_list(raw: Any) -> List[Dict[str, Any]]:
    text = _decode(raw)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _meta_map(client: redis.Redis, key: str) -> Dict[str, str]:
    raw = client.hgetall(key) or {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        dk = _decode(k)
        dv = _decode(v)
        if dk is None or dv is None:
            continue
        out[str(dk)] = str(dv)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Redis v2 document/partition/chunk keys from v1.")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--redis-prefix", default=os.getenv("REDIS_PREFIX", "unstructured"))
    parser.add_argument("--default-collection", default=os.getenv("QDRANT_COLLECTION", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = redis.from_url(args.redis_url)
    hashes_key = f"{args.redis_prefix}:pdf:hashes"
    raw_doc_ids = client.smembers(hashes_key) or set()
    doc_ids = sorted(str(_decode(entry)) for entry in raw_doc_ids if _decode(entry))

    scanned = 0
    updated = 0
    skipped = 0
    for doc_id in doc_ids:
        scanned += 1
        meta_key = f"{args.redis_prefix}:pdf:{doc_id}:meta"
        meta = _meta_map(client, meta_key)
        source = meta.get("source") or ""
        bucket, key = split_source_path(source)
        if not bucket or not key:
            bucket = os.getenv("SOURCE_BUCKET", "local")
            key = source or f"{doc_id}.pdf"

        partitions_key = meta.get("partitions_key") or f"{args.redis_prefix}:pdf:{doc_id}:partitions"
        chunks_key = meta.get("chunks_key") or f"{args.redis_prefix}:pdf:{doc_id}:chunks"
        partitions = _json_list(client.get(partitions_key))
        chunks = _json_list(client.get(chunks_key))
        if not chunks:
            skipped += 1
            continue

        if args.dry_run:
            updated += 1
            continue

        collections_key = meta.get("collections_key") or f"{args.redis_prefix}:pdf:{doc_id}:collections"
        collections = [
            str(_decode(entry))
            for entry in (client.smembers(collections_key) or set())
            if _decode(entry)
        ]
        collection = collections[0] if collections else (args.default_collection or None)
        write_v2_document_payloads(
            redis_client=client,
            prefix=args.redis_prefix,
            doc_hash=doc_id,
            source=SourceDescriptor(bucket=bucket, key=key, version_id=None),
            partitions_payload=partitions,
            chunks_payload=chunks,
            collection=collection,
        )
        updated += 1

    print(
        json.dumps(
            {
                "scanned": scanned,
                "updated": updated,
                "skipped": skipped,
                "dry_run": args.dry_run,
                "redis_prefix": args.redis_prefix,
            }
        )
    )


if __name__ == "__main__":
    main()
