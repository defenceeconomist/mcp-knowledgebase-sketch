import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterable, List
from urllib.parse import unquote_plus

from fastembed import SparseTextEmbedding, TextEmbedding
from minio import Minio
from minio.error import S3Error
from qdrant_client import QdrantClient

from mcp_research.ingest_unstructured import (
    _hash_bytes,
    _redis_key,
    elements_to_chunks,
    load_dotenv,
    load_env_bool,
    load_env_int,
    parse_languages,
    partition_pdf,
    record_collection_mapping,
    upload_to_redis,
)
from mcp_research.upsert_chunks import ensure_collection, upsert_items


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _get_minio_client(endpoint: str, access_key: str, secret_key: str, secure: bool) -> Minio:
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def _decode_redis_value(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _load_env_list(key: str) -> List[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _normalize_events(events: Iterable[str]) -> List[str]:
    return [event.strip() for event in events if event and event.strip()]


def _get_redis_client(redis_url: str):
    if not redis_url:
        return None
    try:
        import redis  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("redis package is required for Redis usage") from exc
    return redis.from_url(redis_url)


class MinioIngestor:
    def __init__(
        self,
        minio_client: Minio,
        qdrant_client: QdrantClient,
        dense_model: TextEmbedding,
        sparse_model: SparseTextEmbedding,
        redis_client,
        redis_prefix: str,
        unstructured_api_key: str,
        unstructured_api_url: str,
        unstructured_strategy: str,
        unstructured_chunking: str | None,
        chunk_size: int,
        chunk_overlap: int,
        languages: List[str] | None,
        skip_existing: bool,
    ) -> None:
        self.minio_client = minio_client
        self.qdrant_client = qdrant_client
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.redis_client = redis_client
        self.redis_prefix = redis_prefix
        self.unstructured_api_key = unstructured_api_key
        self.unstructured_api_url = unstructured_api_url
        self.unstructured_strategy = unstructured_strategy
        self.unstructured_chunking = unstructured_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.languages = languages
        self.skip_existing = skip_existing
        self._collections_ready: set[str] = set()
        self._lock = threading.Lock()
        self._unstructured_client = None

    def _ensure_collection(self, collection: str) -> None:
        if collection in self._collections_ready:
            return
        dense_dim = len(next(iter(self.dense_model.embed(["dimension probe"]))))
        ensure_collection(self.qdrant_client, collection, dense_dim)
        self._collections_ready.add(collection)

    def _load_chunks_from_redis(self, doc_id: str) -> List[dict] | None:
        if not self.redis_client:
            return None
        chunks_key = _redis_key(self.redis_prefix, doc_id, "chunks")
        raw = self.redis_client.get(chunks_key)
        raw = _decode_redis_value(raw)
        if not raw:
            return None
        payload = json.loads(raw)
        return payload if isinstance(payload, list) else None

    def _collection_mapping_exists(self, doc_id: str, collection: str) -> bool:
        if not self.redis_client:
            return False
        collections_key = f"{self.redis_prefix}:pdf:{doc_id}:collections"
        return bool(self.redis_client.sismember(collections_key, collection))

    def _get_unstructured_client(self):
        if not self.unstructured_api_key:
            raise RuntimeError("UNSTRUCTURED_API_KEY is required to partition files")
        if self._unstructured_client is None:
            from unstructured_client import (  # pylint: disable=import-outside-toplevel
                UnstructuredClient,
            )

            self._unstructured_client = UnstructuredClient(
                api_key_auth=self.unstructured_api_key,
                server_url=self.unstructured_api_url,
            )
        return self._unstructured_client

    def _partition_bytes(self, object_name: str, file_bytes: bytes) -> List[dict]:
        client = self._get_unstructured_client()
        elements = partition_pdf(
            client,
            pdf_path=Path(object_name),
            file_bytes=file_bytes,
            strategy=self.unstructured_strategy,
            chunking_strategy=self.unstructured_chunking,
            max_characters=self.chunk_size,
            overlap=self.chunk_overlap,
            languages=self.languages,
        )
        return [element.to_dict() if hasattr(element, "to_dict") else element for element in elements]

    def process_object(self, bucket: str, object_name: str) -> None:
        object_name = unquote_plus(object_name)
        if not object_name:
            return
        if not object_name.lower().endswith(".pdf"):
            logger.info("Skipping non-PDF upload: %s/%s", bucket, object_name)
            return
        source = f"{bucket}/{object_name}"
        logger.info("Processing s3://%s", source)

        response = self.minio_client.get_object(bucket, object_name)
        try:
            file_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        doc_id = _hash_bytes(file_bytes)
        if self.skip_existing and self._collection_mapping_exists(doc_id, bucket):
            logger.info("Skipping %s (already mapped to %s)", source, bucket)
            return

        chunk_items = self._load_chunks_from_redis(doc_id)
        if not chunk_items:
            elements_payload = self._partition_bytes(object_name, file_bytes)
            chunk_payloads, page_ranges = elements_to_chunks(elements_payload)
            if not chunk_payloads:
                logger.warning("No text extracted from %s", source)
                return
            chunk_items = []
            for idx, (chunk, page_list) in enumerate(zip(chunk_payloads, page_ranges)):
                chunk_items.append(
                    {
                        "document_id": doc_id,
                        "source": source,
                        "chunk_index": idx,
                        "pages": page_list,
                        "text": chunk,
                    }
                )
            if self.redis_client:
                upload_to_redis(
                    redis_client=self.redis_client,
                    doc_id=doc_id,
                    source=source,
                    partitions_payload=elements_payload,
                    chunks_payload=chunk_items,
                    prefix=self.redis_prefix,
                    collection=bucket,
                )

        with self._lock:
            self._ensure_collection(bucket)
            upsert_items(
                client=self.qdrant_client,
                collection=bucket,
                items=chunk_items,
                dense_model=self.dense_model,
                sparse_model=self.sparse_model,
                batch_size=64,
            )

        if self.redis_client:
            record_collection_mapping(
                redis_client=self.redis_client,
                doc_id=doc_id,
                collection=bucket,
                prefix=self.redis_prefix,
            )
        logger.info("Upserted %d chunks into '%s'", len(chunk_items), bucket)


def _listen_bucket(
    bucket: str,
    minio_client: Minio,
    ingestor: MinioIngestor,
    prefix: str,
    suffix: str,
    events: List[str],
) -> None:
    logger.info("Listening for %s in bucket %s", events, bucket)
    try:
        for event in minio_client.listen_bucket_notification(
            bucket_name=bucket,
            prefix=prefix,
            suffix=suffix,
            events=events,
        ):
            if not event:
                continue
            for record in event.get("Records", []):
                s3_info = record.get("s3", {})
                bucket_name = s3_info.get("bucket", {}).get("name") or bucket
                object_name = s3_info.get("object", {}).get("key")
                if not object_name:
                    continue
                try:
                    ingestor.process_object(bucket_name, object_name)
                except Exception as exc:
                    logger.exception("Failed to process %s/%s: %s", bucket_name, object_name, exc)
    except S3Error as exc:
        logger.error("MinIO listen error for %s: %s", bucket, exc)


def main() -> None:
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Listen for MinIO uploads and ingest into Qdrant collections.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help="Bucket to listen on (repeatable).",
    )
    parser.add_argument(
        "--all-buckets",
        action="store_true",
        help="Listen on all buckets available at startup.",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("MINIO_PREFIX", ""),
        help="Filter notifications by prefix.",
    )
    parser.add_argument(
        "--suffix",
        default=os.getenv("MINIO_SUFFIX", ".pdf"),
        help="Filter notifications by suffix.",
    )
    parser.add_argument(
        "--events",
        default=",".join(_load_env_list("MINIO_EVENTS") or ["s3:ObjectCreated:*"]),
        help="Comma-separated list of MinIO events to subscribe to.",
    )
    args = parser.parse_args()

    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    secure = load_env_bool("MINIO_SECURE", False)

    if not access_key or not secret_key:
        raise RuntimeError("MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required")

    minio_client = _get_minio_client(endpoint, access_key, secret_key, secure)

    buckets = list(args.bucket)
    if not buckets:
        buckets = _load_env_list("MINIO_BUCKETS")
    watch_all = args.all_buckets or load_env_bool("MINIO_ALL_BUCKETS", False)
    if not buckets and watch_all:
        buckets = [bucket.name for bucket in minio_client.list_buckets()]
    if not buckets and not watch_all:
        raise RuntimeError("No buckets specified. Use --bucket or MINIO_BUCKETS.")

    redis_url = os.getenv("REDIS_URL", "")
    redis_prefix = os.getenv("REDIS_PREFIX", "unstructured")
    redis_client = _get_redis_client(redis_url) if redis_url else None

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    dense_model = TextEmbedding(model_name=os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5"))
    sparse_model = SparseTextEmbedding(model_name=os.getenv("SPARSE_MODEL", "Qdrant/bm25"))
    qdrant_client = QdrantClient(url=qdrant_url)

    unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY", "")
    unstructured_api_url = os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructured.io")
    unstructured_strategy = os.getenv("UNSTRUCTURED_STRATEGY", "hi_res")
    chunking_strategy_raw = os.getenv("UNSTRUCTURED_CHUNKING_STRATEGY", "basic")
    unstructured_chunking = (
        chunking_strategy_raw if chunking_strategy_raw.lower() != "none" else None
    )
    chunk_size = load_env_int("CHUNK_SIZE", 1200)
    chunk_overlap = load_env_int("CHUNK_OVERLAP", 200)
    languages = parse_languages(os.getenv("UNSTRUCTURED_LANGUAGES"))
    skip_existing = load_env_bool("MINIO_SKIP_EXISTING", True)

    ingestor = MinioIngestor(
        minio_client=minio_client,
        qdrant_client=qdrant_client,
        dense_model=dense_model,
        sparse_model=sparse_model,
        redis_client=redis_client,
        redis_prefix=redis_prefix,
        unstructured_api_key=unstructured_api_key,
        unstructured_api_url=unstructured_api_url,
        unstructured_strategy=unstructured_strategy,
        unstructured_chunking=unstructured_chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        languages=languages,
        skip_existing=skip_existing,
    )

    events = _normalize_events(args.events.split(","))
    active_buckets: set[str] = set()
    lock = threading.Lock()

    def start_listener(bucket_name: str) -> None:
        client = _get_minio_client(endpoint, access_key, secret_key, secure)
        thread = threading.Thread(
            target=_listen_bucket,
            args=(bucket_name, client, ingestor, args.prefix, args.suffix, events),
            daemon=True,
        )
        thread.start()

    def ensure_listeners(bucket_names: Iterable[str]) -> None:
        with lock:
            for bucket_name in bucket_names:
                if bucket_name in active_buckets:
                    continue
                start_listener(bucket_name)
                active_buckets.add(bucket_name)

    ensure_listeners(buckets)
    logger.info("MinIO ingest listeners running for: %s", sorted(active_buckets))
    refresh_seconds = load_env_int("MINIO_BUCKET_REFRESH_SECONDS", 30)
    try:
        while True:
            time.sleep(refresh_seconds)
            if not watch_all:
                continue
            try:
                latest = [bucket.name for bucket in minio_client.list_buckets()]
            except S3Error as exc:
                logger.warning("Failed to list buckets: %s", exc)
                continue
            ensure_listeners(latest)
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
