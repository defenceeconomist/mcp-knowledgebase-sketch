import argparse
import json
import logging
import os
from pathlib import Path

from mcp_research.ingest_unstructured import _get_redis_client, _redis_key, _source_key, load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _decode(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _source_from_chunks(redis_client, prefix: str, doc_id: str) -> str | None:
    chunks_key = _redis_key(prefix, doc_id, "chunks")
    raw = _decode(redis_client.get(chunks_key))
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        return None
    source = first.get("source")
    return source if isinstance(source, str) and source else None


def main() -> None:
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Backfill Redis source index for existing documents.",
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
    missing_source = 0
    for raw_id in doc_ids:
        doc_id = _decode(raw_id)
        if not doc_id:
            continue
        total += 1
        meta_key = _redis_key(args.redis_prefix, doc_id, "meta")
        source = _decode(redis_client.hget(meta_key, "source"))
        if not source:
            source = _source_from_chunks(redis_client, args.redis_prefix, doc_id)
        if not source:
            missing_source += 1
            continue
        source_key = _source_key(args.redis_prefix, source)
        if args.dry_run:
            logger.info("Would add %s -> %s", source_key, doc_id)
        else:
            redis_client.sadd(source_key, doc_id)
            updated += 1

    logger.info(
        "Backfill complete: total=%d updated=%d missing_source=%d",
        total,
        updated,
        missing_source,
    )


if __name__ == "__main__":
    main()
