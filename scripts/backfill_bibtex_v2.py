#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import redis

from mcp_research.schema_v2 import bibtex_v2_doc_key, bibtex_v2_source_doc_key, source_id


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _json_map(raw: Any) -> Dict[str, Any]:
    text = _decode(raw)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_doc_ids(client: redis.Redis, prefix: str, source: str) -> List[str]:
    source_key = f"{prefix}:pdf:source:{source}"
    values: List[str] = []
    if hasattr(client, "smembers"):
        for entry in client.smembers(source_key) or set():
            decoded = _decode(entry)
            if decoded:
                values.append(str(decoded))
    if not values:
        raw = _decode(client.get(source_key))
        if raw:
            values.append(str(raw))
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill BibTeX v2 keys from v1 file keys.")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--bibtex-prefix", default=os.getenv("BIBTEX_REDIS_PREFIX", "bibtex"))
    parser.add_argument(
        "--source-prefix",
        default=os.getenv("BIBTEX_SOURCE_REDIS_PREFIX", os.getenv("REDIS_PREFIX", "unstructured")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = redis.from_url(args.redis_url)
    files_key = f"{args.bibtex_prefix}:files"
    raw_files = client.smembers(files_key) or set()
    files = sorted(str(_decode(entry)) for entry in raw_files if _decode(entry))

    scanned = 0
    updated = 0
    skipped = 0
    for file_id in files:
        scanned += 1
        if "/" not in file_id:
            skipped += 1
            continue
        bucket, object_name = file_id.split("/", 1)
        v1_key = f"{args.bibtex_prefix}:file:{bucket}/{object_name}"
        metadata = _json_map(client.get(v1_key))
        if not metadata:
            skipped += 1
            continue

        doc_ids = _source_doc_ids(client, args.source_prefix, f"{bucket}/{object_name}")
        if not doc_ids:
            skipped += 1
            continue
        doc_id = doc_ids[0]
        sid = source_id(bucket, object_name, None)
        if not args.dry_run:
            client.set(bibtex_v2_doc_key(args.bibtex_prefix, doc_id), json.dumps(metadata, ensure_ascii=True))
            client.set(bibtex_v2_source_doc_key(args.bibtex_prefix, sid), doc_id)
        updated += 1

    print(
        json.dumps(
            {
                "scanned": scanned,
                "updated": updated,
                "skipped": skipped,
                "dry_run": args.dry_run,
                "bibtex_prefix": args.bibtex_prefix,
            }
        )
    )


if __name__ == "__main__":
    main()
