import argparse
import json
import logging
import os
from pathlib import Path

from mcp_research.ingest_unstructured import _get_redis_client, _redis_key, load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _decode(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _collections_key(prefix: str, doc_id: str) -> str:
    return f"{prefix}:pdf:{doc_id}:collections"


def _extract_collections_from_chunks(chunks_payload) -> set[str]:
    collections: set[str] = set()
    if not isinstance(chunks_payload, list):
        return collections
    for entry in chunks_payload:
        if not isinstance(entry, dict):
            continue
        bucket = entry.get("bucket") or entry.get("collection")
        if bucket:
            collections.add(bucket)
            continue
        source = entry.get("source")
        if isinstance(source, str) and "/" in source:
            collections.add(source.split("/", 1)[0])
    return collections


def main() -> None:
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Backfill Redis collection tags + collections_key for existing documents.",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", ""),
        help="Redis URL (required)",
    )
    parser.add_argument(
        "--redis-prefix",
        default=os.getenv("REDIS_PREFIX", "unstructured"),
        help="Redis key prefix for payloads",
    )
    parser.add_argument(
        "--fallback-collection",
        default=os.getenv("BACKFILL_COLLECTION", ""),
        help="Collection to use when no bucket is present in chunk payloads",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log updates without writing to Redis",
    )
    args = parser.parse_args()

    redis_client = _get_redis_client(args.redis_url)
    if not redis_client:
        raise RuntimeError("REDIS_URL is required and redis must be installed")

    hashes_key = f"{args.redis_prefix}:pdf:hashes"
    doc_ids = redis_client.smembers(hashes_key)
    if not doc_ids:
        logger.info("No document ids found in %s", hashes_key)
        return

    total = 0
    updated = 0
    missing_collections = 0
    for raw_id in doc_ids:
        doc_id = _decode(raw_id)
        if not doc_id:
            continue
        total += 1
        chunks_key = _redis_key(args.redis_prefix, doc_id, "chunks")
        raw_chunks = _decode(redis_client.get(chunks_key))
        if not raw_chunks:
            missing_collections += 1
            continue
        try:
            chunks_payload = json.loads(raw_chunks)
        except json.JSONDecodeError:
            missing_collections += 1
            continue
        collections = _extract_collections_from_chunks(chunks_payload)
        if not collections and args.fallback_collection:
            collections = {args.fallback_collection}
        if not collections:
            missing_collections += 1
            continue

        collections_key = _collections_key(args.redis_prefix, doc_id)
        if args.dry_run:
            logger.info("Would set %s -> %s", collections_key, sorted(collections))
            continue

        for collection in collections:
            redis_client.sadd(collections_key, collection)
        meta_key = _redis_key(args.redis_prefix, doc_id, "meta")
        redis_client.hset(meta_key, mapping={"collections_key": collections_key})
        updated += 1

    logger.info(
        "Backfill complete: total=%d updated=%d missing_collections=%d",
        total,
        updated,
        missing_collections,
    )


if __name__ == "__main__":
    main()
