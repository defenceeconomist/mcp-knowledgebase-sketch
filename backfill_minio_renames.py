import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote_plus

from minio import Minio
from minio.error import S3Error
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from mcp_research.ingest_unstructured import _hash_bytes, load_dotenv
from mcp_research.link_resolver import build_source_ref


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _decode_redis_value(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _decode_redis_hash(raw: Dict) -> Dict:
    return {_decode_redis_value(k): _decode_redis_value(v) for k, v in (raw or {}).items()}


def _split_source(source: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not source or "/" not in source:
        return None, None
    bucket, key = source.split("/", 1)
    return bucket or None, key or None


def _source_key(prefix: str, source: str) -> str:
    return f"{prefix}:pdf:source:{source}"


def _get_minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    secure = os.getenv("MINIO_SECURE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    if not access_key or not secret_key:
        raise RuntimeError("MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _get_redis_client(redis_url: str):
    if not redis_url:
        return None
    try:
        import redis  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("redis package is required for Redis usage") from exc
    return redis.from_url(redis_url)


def _load_env_list(key: str) -> List[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _list_objects(minio_client: Minio, bucket: str, prefix: str, suffix: str) -> Iterable[str]:
    for entry in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
        name = entry.object_name or ""
        if not name:
            continue
        if suffix and not name.lower().endswith(suffix.lower()):
            continue
        yield name


def _pages_to_range(pages) -> Tuple[Optional[int], Optional[int]]:
    if not pages:
        return None, None
    try:
        return min(pages), max(pages)
    except (TypeError, ValueError):
        return None, None


def _update_chunks_payload(
    chunks_payload: List[dict],
    new_source: str,
    bucket: str,
    key: str,
) -> List[dict]:
    for entry in chunks_payload:
        if not isinstance(entry, dict):
            continue
        page_start = entry.get("page_start")
        page_end = entry.get("page_end")
        if page_start is None or page_end is None:
            range_start, range_end = _pages_to_range(entry.get("pages") or [])
            if page_start is None:
                page_start = range_start
                entry["page_start"] = range_start
            if page_end is None:
                page_end = range_end
                entry["page_end"] = range_end
        version_id = entry.get("version_id")
        entry.update(
            {
                "source": new_source,
                "bucket": bucket,
                "key": key,
                "source_ref": build_source_ref(
                    bucket=bucket,
                    key=key,
                    page_start=page_start,
                    page_end=page_end,
                    version_id=version_id,
                ),
            }
        )
    return chunks_payload


def _update_qdrant_payloads(
    client: QdrantClient,
    collection: str,
    doc_id: str,
    new_source: str,
    bucket: str,
    key: str,
    dry_run: bool,
    batch_size: int,
) -> int:
    updated = 0
    next_page = None
    qdrant_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
    )
    while True:
        points, next_page = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
            scroll_filter=qdrant_filter,
        )
        if not points:
            break
        for point in points:
            payload = point.payload or {}
            page_start = payload.get("page_start")
            page_end = payload.get("page_end")
            if page_start is None or page_end is None:
                range_start, range_end = _pages_to_range(payload.get("pages") or [])
                if page_start is None:
                    page_start = range_start
                if page_end is None:
                    page_end = range_end
            version_id = payload.get("version_id")
            patch = {
                "source": new_source,
                "bucket": bucket,
                "key": key,
                "source_ref": build_source_ref(
                    bucket=bucket,
                    key=key,
                    page_start=page_start,
                    page_end=page_end,
                    version_id=version_id,
                ),
            }
            if dry_run:
                updated += 1
                continue
            client.set_payload(
                collection_name=collection,
                payload=patch,
                points=[point.id],
            )
            updated += 1
        if next_page is None:
            break
    return updated


def main() -> None:
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Backfill Redis/Qdrant metadata after MinIO renames by content hash.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help="Bucket to scan (repeatable).",
    )
    parser.add_argument(
        "--all-buckets",
        action="store_true",
        help="Scan all buckets.",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("MINIO_PREFIX", ""),
        help="Only scan objects with this prefix.",
    )
    parser.add_argument(
        "--suffix",
        default=os.getenv("MINIO_SUFFIX", ".pdf"),
        help="Only scan objects with this suffix.",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", ""),
        help="Redis URL.",
    )
    parser.add_argument(
        "--redis-prefix",
        default=os.getenv("REDIS_PREFIX", "unstructured"),
        help="Redis key prefix.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Qdrant scroll batch size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log updates without writing to Redis or Qdrant.",
    )
    args = parser.parse_args()

    minio_client = _get_minio_client()
    redis_client = _get_redis_client(args.redis_url)
    if not redis_client:
        raise RuntimeError("REDIS_URL is required to update Redis metadata")
    qdrant_client = QdrantClient(url=args.qdrant_url)

    buckets = list(args.bucket)
    if not buckets:
        buckets = _load_env_list("MINIO_BUCKETS")
    if args.all_buckets or os.getenv("MINIO_ALL_BUCKETS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        buckets = [bucket.name for bucket in minio_client.list_buckets()]
    if not buckets:
        raise RuntimeError("No buckets specified. Use --bucket or MINIO_BUCKETS.")

    seen: Dict[str, str] = {}
    conflicts: Dict[str, List[str]] = {}
    scanned = 0
    redis_updates = 0
    qdrant_updates = 0

    for bucket in buckets:
        try:
            object_names = list(_list_objects(minio_client, bucket, args.prefix, args.suffix))
        except S3Error as exc:
            logger.error("Failed to list objects for %s: %s", bucket, exc)
            continue

        for object_name in object_names:
            scanned += 1
            object_name = unquote_plus(object_name)
            source = f"{bucket}/{object_name}"
            try:
                response = minio_client.get_object(bucket, object_name)
                try:
                    data = response.read()
                finally:
                    response.close()
                    response.release_conn()
            except S3Error as exc:
                logger.warning("Failed to download %s: %s", source, exc)
                continue

            doc_id = _hash_bytes(data)
            if doc_id in seen and seen[doc_id] != source:
                conflicts.setdefault(doc_id, []).append(source)
                logger.warning(
                    "Skipping %s (doc_id %s already seen at %s)",
                    source,
                    doc_id,
                    seen[doc_id],
                )
                continue
            seen[doc_id] = source

            meta_key = f"{args.redis_prefix}:pdf:{doc_id}:meta"
            if not redis_client.exists(meta_key):
                continue
            meta = _decode_redis_hash(redis_client.hgetall(meta_key))
            old_source = meta.get("source")
            if old_source == source:
                continue

            old_bucket, old_key = _split_source(old_source)
            if not old_bucket:
                old_bucket = bucket

            logger.info("Updating %s -> %s (doc_id=%s)", old_source, source, doc_id)

            if not args.dry_run:
                redis_client.hset(meta_key, mapping={"source": source})
                if old_source:
                    old_source_key = _source_key(args.redis_prefix, old_source)
                    redis_client.srem(old_source_key, doc_id)
                    if redis_client.scard(old_source_key) == 0:
                        redis_client.delete(old_source_key)
                redis_client.sadd(_source_key(args.redis_prefix, source), doc_id)

            chunks_key = meta.get("chunks_key") or f"{args.redis_prefix}:pdf:{doc_id}:chunks"
            raw_chunks = _decode_redis_value(redis_client.get(chunks_key))
            if raw_chunks:
                try:
                    chunks_payload = json.loads(raw_chunks)
                except json.JSONDecodeError:
                    chunks_payload = None
                if isinstance(chunks_payload, list):
                    updated_chunks = _update_chunks_payload(
                        chunks_payload=chunks_payload,
                        new_source=source,
                        bucket=bucket,
                        key=object_name,
                    )
                    if not args.dry_run:
                        redis_client.set(
                            chunks_key,
                            json.dumps(updated_chunks, ensure_ascii=True),
                        )
            redis_updates += 1

            if old_bucket != bucket:
                logger.warning(
                    "Document moved across buckets (%s -> %s); payloads updated in old collection only.",
                    old_bucket,
                    bucket,
                )

            if qdrant_client.collection_exists(old_bucket):
                qdrant_updates += _update_qdrant_payloads(
                    client=qdrant_client,
                    collection=old_bucket,
                    doc_id=doc_id,
                    new_source=source,
                    bucket=bucket,
                    key=object_name,
                    dry_run=args.dry_run,
                    batch_size=args.batch_size,
                )
            else:
                logger.warning("Qdrant collection %s not found; skipping", old_bucket)

    logger.info(
        "Backfill complete: scanned=%d redis_updates=%d qdrant_updates=%d",
        scanned,
        redis_updates,
        qdrant_updates,
    )
    if conflicts:
        logger.warning("Conflicts detected for %d document_ids (duplicates in MinIO).", len(conflicts))


if __name__ == "__main__":
    main()
