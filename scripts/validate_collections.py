import argparse
import json
import logging
import os
from pathlib import Path

from mcp_research.ingest_unstructured import _get_redis_client, _redis_key, load_dotenv
from mcp_research.runtime_utils import decode_redis_value as _decode


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _collections_key(prefix: str, doc_id: str) -> str:
    return f"{prefix}:pdf:{doc_id}:collections"


def _has_collection_tag(chunks_payload) -> bool:
    if not isinstance(chunks_payload, list):
        return False
    for entry in chunks_payload:
        if not isinstance(entry, dict):
            continue
        if entry.get("bucket") or entry.get("collection"):
            return True
        source = entry.get("source")
        if isinstance(source, str) and "/" in source:
            return True
    return False


def main() -> None:
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Validate Redis collection coverage for existing documents.",
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
        "--limit",
        type=int,
        default=20,
        help="Max doc_ids to print per missing category",
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

    missing_collections_key = []
    missing_collection_tags = []
    missing_chunks = []

    for raw_id in doc_ids:
        doc_id = _decode(raw_id)
        if not doc_id:
            continue
        collections_key = _collections_key(args.redis_prefix, doc_id)
        if redis_client.scard(collections_key) == 0:
            missing_collections_key.append(doc_id)

        chunks_key = _redis_key(args.redis_prefix, doc_id, "chunks")
        raw_chunks = _decode(redis_client.get(chunks_key))
        if not raw_chunks:
            missing_chunks.append(doc_id)
            continue
        try:
            chunks_payload = json.loads(raw_chunks)
        except json.JSONDecodeError:
            missing_chunks.append(doc_id)
            continue
        if not _has_collection_tag(chunks_payload):
            missing_collection_tags.append(doc_id)

    def _sample(items):
        return items[: args.limit]

    logger.info("Documents scanned: %d", len(doc_ids))
    logger.info("Missing collections_key: %d", len(missing_collections_key))
    logger.info("Missing collection tags in chunks: %d", len(missing_collection_tags))
    logger.info("Missing chunks payloads: %d", len(missing_chunks))
    if missing_collections_key:
        logger.info("Sample missing collections_key: %s", _sample(missing_collections_key))
    if missing_collection_tags:
        logger.info("Sample missing collection tags: %s", _sample(missing_collection_tags))
    if missing_chunks:
        logger.info("Sample missing chunks: %s", _sample(missing_chunks))


if __name__ == "__main__":
    main()
