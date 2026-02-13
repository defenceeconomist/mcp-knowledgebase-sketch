#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

from qdrant_client import QdrantClient

from mcp_research.schema_v2 import chunk_hash, partition_hash, source_id, split_source_path


def _target_collections(client: QdrantClient, collection: str | None, all_collections: bool) -> List[str]:
    if all_collections:
        payload = client.get_collections()
        names = sorted([entry.name for entry in payload.collections])
        if collection:
            return sorted(set(names + [collection]))
        return names
    if collection:
        return [collection]
    return [os.getenv("QDRANT_COLLECTION", "pdf_chunks")]


def _build_patch(payload: Dict[str, Any], collection: str, point_id: Any) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    doc_hash = str(payload.get("doc_hash") or payload.get("document_id") or "").strip()
    if not doc_hash:
        doc_hash = hashlib.sha256(f"{collection}|{point_id}".encode("utf-8")).hexdigest()
        patch["doc_hash"] = doc_hash
    if doc_hash and not payload.get("doc_hash"):
        patch["doc_hash"] = doc_hash

    if doc_hash:
        if not payload.get("partition_hash"):
            patch["partition_hash"] = partition_hash(doc_hash, payload)
        if not payload.get("chunk_hash"):
            patch["chunk_hash"] = chunk_hash(doc_hash, payload)

    bucket = payload.get("bucket")
    key = payload.get("key")
    if (not bucket or not key) and payload.get("source"):
        source_bucket, source_key = split_source_path(str(payload.get("source")))
        bucket = bucket or source_bucket
        key = key or source_key

    if bucket and key and not payload.get("source_id"):
        patch["source_id"] = source_id(str(bucket), str(key), payload.get("version_id"))
    elif not payload.get("source_id"):
        patch["source_id"] = hashlib.sha1(f"{collection}|{point_id}".encode("utf-8")).hexdigest()

    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Qdrant payload with v2 hash fields.")
    parser.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "pdf_chunks"))
    parser.add_argument("--all-collections", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=args.url)
    collections = _target_collections(client, args.collection, args.all_collections)

    total_scanned = 0
    total_updated = 0
    for collection in collections:
        if not client.collection_exists(collection):
            continue
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=max(1, args.batch_size),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                total_scanned += 1
                payload = point.payload or {}
                patch = _build_patch(payload, collection=collection, point_id=point.id)
                if not patch:
                    continue
                if not args.dry_run:
                    client.set_payload(
                        collection_name=collection,
                        payload=patch,
                        points=[point.id],
                    )
                total_updated += 1
            if offset is None:
                break

    print(
        json.dumps(
            {
                "collections": collections,
                "scanned": total_scanned,
                "updated": total_updated,
                "dry_run": args.dry_run,
            }
        )
    )


if __name__ == "__main__":
    main()
