import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient

from mcp_research.link_resolver import build_source_ref


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _split_source(source: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not source or "/" not in source:
        return None, None
    bucket, key = source.split("/", 1)
    return bucket or None, key or None


def _pages_to_range(pages) -> Tuple[Optional[int], Optional[int]]:
    if not pages:
        return None, None
    try:
        return min(pages), max(pages)
    except (TypeError, ValueError):
        return None, None


def _build_missing_payload(payload: Dict) -> Dict:
    missing: Dict = {}
    bucket = payload.get("bucket")
    key = payload.get("key")
    if not bucket or not key:
        bucket_from_source, key_from_source = _split_source(payload.get("source"))
        if not bucket and bucket_from_source:
            bucket = bucket_from_source
            missing["bucket"] = bucket_from_source
        if not key and key_from_source:
            key = key_from_source
            missing["key"] = key_from_source

    page_start = payload.get("page_start")
    page_end = payload.get("page_end")
    if page_start is None or page_end is None:
        pages = payload.get("pages") or []
        range_start, range_end = _pages_to_range(pages)
        if page_start is None and range_start is not None:
            page_start = range_start
            missing["page_start"] = range_start
        if page_end is None and range_end is not None:
            page_end = range_end
            missing["page_end"] = range_end

    if not payload.get("source_ref") and bucket and key:
        missing["source_ref"] = build_source_ref(
            bucket=bucket,
            key=key,
            page_start=page_start,
            page_end=page_end,
            version_id=payload.get("version_id"),
        )

    if payload.get("document_id") and payload.get("chunk_index") is not None:
        chunk_id = payload.get("chunk_id")
        if not chunk_id:
            missing["chunk_id"] = f"{payload['document_id']}:{payload['chunk_index']}"

    return missing


def backfill_collection(
    client: QdrantClient,
    collection: str,
    batch_size: int,
    dry_run: bool,
) -> None:
    next_page = None
    updated = 0
    scanned = 0

    while True:
        points, next_page = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            scanned += 1
            payload = point.payload or {}
            missing = _build_missing_payload(payload)
            if not missing:
                continue
            if dry_run:
                logger.info("Would update %s: %s", point.id, missing)
                updated += 1
                continue
            client.set_payload(
                collection_name=collection,
                payload=missing,
                points=[point.id],
            )
            updated += 1
        if next_page is None:
            break

    logger.info("Backfill complete: scanned=%d updated=%d", scanned, updated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Qdrant payload metadata (bucket/key/pages/source_ref/chunk_id).",
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
        "--batch-size",
        type=int,
        default=256,
        help="Scroll batch size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log updates without writing to Qdrant",
    )
    args = parser.parse_args()

    client = QdrantClient(url=args.url)
    backfill_collection(
        client=client,
        collection=args.collection,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
