import argparse
import hashlib
import logging
import os
from typing import Any, Dict, Iterable, List, Tuple

from qdrant_client import QdrantClient, models


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("-").isdigit():
            return int(text)
    return None


def _sorted_pages(payload: Dict[str, Any]) -> tuple[int, ...]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return ()
    values = [_coerce_int(value) for value in pages]
    return tuple(sorted({value for value in values if value is not None}))


def _extract_text(payload: Dict[str, Any]) -> str:
    for key in ("text", "chunk_text", "content", "chunk", "page_content", "raw_text"):
        value = payload.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return ""


def _text_fingerprint(payload: Dict[str, Any]) -> str:
    text = _extract_text(payload)
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _doc_identity(payload: Dict[str, Any]) -> str:
    document_id = payload.get("doc_hash") or payload.get("document_id")
    sid = payload.get("source_id")
    source = payload.get("source")
    bucket = payload.get("bucket")
    key = payload.get("key")
    if document_id:
        return f"doc:{document_id}"
    if sid:
        return f"source_id:{sid}"
    if source:
        return f"source:{source}"
    if bucket and key:
        return f"object:{bucket}/{key}"
    return ""


def _identity_key(payload: Dict[str, Any], point_id: Any) -> Tuple[Any, ...]:
    chunk_id = payload.get("chunk_id")
    if chunk_id:
        return ("chunk_id", str(chunk_id))

    pages = _sorted_pages(payload)
    chunk_index = _coerce_int(payload.get("chunk_index"))
    page_start = _coerce_int(payload.get("page_start"))
    page_end = _coerce_int(payload.get("page_end"))
    source_ref = str(payload.get("source_ref") or "")
    text_hash = _text_fingerprint(payload)
    doc_key = _doc_identity(payload)

    has_payload_identity = any(
        (
            chunk_index is not None,
            page_start is not None,
            page_end is not None,
            bool(pages),
            bool(source_ref),
            bool(text_hash),
            bool(doc_key),
        )
    )
    if not has_payload_identity:
        return ("point", str(point_id))

    return (
        "payload",
        doc_key,
        chunk_index,
        page_start,
        page_end,
        pages,
        source_ref,
        text_hash,
    )


def _batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


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


def find_duplicate_point_ids(
    client: QdrantClient,
    collection_name: str,
    batch_size: int,
    max_examples: int,
) -> Dict[str, Any]:
    seen: Dict[Tuple[Any, ...], Any] = {}
    duplicate_ids: List[Any] = []
    examples: List[Dict[str, str]] = []

    scanned = 0
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            scanned += 1
            payload = point.payload or {}
            dedupe_key = _identity_key(payload, point.id)
            if dedupe_key in seen:
                duplicate_ids.append(point.id)
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "keep_id": str(seen[dedupe_key]),
                            "drop_id": str(point.id),
                        }
                    )
            else:
                seen[dedupe_key] = point.id

        if next_offset is None:
            break

    unique_points = len(seen)
    duplicate_points = len(duplicate_ids)
    return {
        "collection": collection_name,
        "scanned": scanned,
        "unique": unique_points,
        "duplicates": duplicate_points,
        "duplicate_ids": duplicate_ids,
        "examples": examples,
    }


def delete_duplicate_point_ids(
    client: QdrantClient,
    collection_name: str,
    point_ids: List[Any],
    delete_batch_size: int,
) -> int:
    deleted = 0
    for batch in _batched(point_ids, max(1, delete_batch_size)):
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=batch),
            wait=True,
        )
        deleted += len(batch)
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove duplicate chunk points from Qdrant collections.")
    parser.add_argument(
        "--url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "pdf_chunks"),
        help="Target collection name (ignored when --all-collections is set unless for explicit inclusion)",
    )
    parser.add_argument(
        "--all-collections",
        action="store_true",
        help="Scan and dedupe all collections",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Scroll batch size while scanning points",
    )
    parser.add_argument(
        "--delete-batch-size",
        type=int,
        default=256,
        help="Delete batch size",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="How many duplicate keep/drop examples to log per collection",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete duplicate points (default is dry-run)",
    )
    args = parser.parse_args()

    client = QdrantClient(url=args.url)
    targets = _target_collections(client, args.collection, args.all_collections)
    if not targets:
        logger.info("No collections found.")
        return

    total_scanned = 0
    total_duplicates = 0
    total_deleted = 0

    for collection_name in targets:
        if not client.collection_exists(collection_name):
            logger.warning("Skipping missing collection: %s", collection_name)
            continue

        result = find_duplicate_point_ids(
            client=client,
            collection_name=collection_name,
            batch_size=max(1, args.batch_size),
            max_examples=max(0, args.max_examples),
        )
        total_scanned += result["scanned"]
        total_duplicates += result["duplicates"]

        logger.info(
            "[%s] scanned=%d unique=%d duplicates=%d",
            collection_name,
            result["scanned"],
            result["unique"],
            result["duplicates"],
        )
        for example in result["examples"]:
            logger.info(
                "[%s] duplicate keep=%s drop=%s",
                collection_name,
                example["keep_id"],
                example["drop_id"],
            )

        if args.apply and result["duplicate_ids"]:
            deleted = delete_duplicate_point_ids(
                client=client,
                collection_name=collection_name,
                point_ids=result["duplicate_ids"],
                delete_batch_size=max(1, args.delete_batch_size),
            )
            total_deleted += deleted
            logger.info("[%s] deleted=%d", collection_name, deleted)

    if args.apply:
        logger.info(
            "Done. scanned=%d duplicates=%d deleted=%d",
            total_scanned,
            total_duplicates,
            total_deleted,
        )
    else:
        logger.info(
            "Dry-run complete. scanned=%d duplicates=%d (no deletions applied)",
            total_scanned,
            total_duplicates,
        )


if __name__ == "__main__":
    main()
