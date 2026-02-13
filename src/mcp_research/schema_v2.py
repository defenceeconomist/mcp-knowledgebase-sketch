from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _env_mode(key: str, default: str, allowed: Iterable[str]) -> str:
    raw = str(os.getenv(key, default)).strip().lower()
    allowed_set = {entry.strip().lower() for entry in allowed}
    if raw in allowed_set:
        return raw
    return default


def redis_schema_write_mode() -> str:
    # Redis document payloads are v2-only after schema migration.
    return _env_mode("REDIS_SCHEMA_WRITE_MODE", "v2", {"v2"})


def redis_schema_read_mode() -> str:
    # Redis document payloads are v2-only after schema migration.
    return _env_mode("REDIS_SCHEMA_READ_MODE", "v2", {"v2"})


def qdrant_payload_schema_mode() -> str:
    # Qdrant payloads are v2-only after schema migration.
    return _env_mode("QDRANT_PAYLOAD_SCHEMA", "v2", {"v2"})


def qdrant_point_id_mode() -> str:
    return _env_mode("QDRANT_POINT_ID_MODE", "uuid", {"uuid", "deterministic"})


def bibtex_schema_write_mode() -> str:
    return _env_mode("BIBTEX_SCHEMA_WRITE_MODE", "v1", {"v1", "dual", "v2"})


def bibtex_schema_read_mode() -> str:
    return _env_mode("BIBTEX_SCHEMA_READ_MODE", "v1", {"v1", "prefer_v2", "v2"})


def search_redis_enrichment_mode() -> str:
    return _env_mode("SEARCH_REDIS_ENRICHMENT", "off", {"off", "partition", "document", "both"})


def search_link_mode() -> str:
    return _env_mode("SEARCH_LINK_MODE", "legacy", {"legacy", "dual"})


def should_write_v1(mode: str) -> bool:
    return mode in {"v1", "dual"}


def should_write_v2(mode: str) -> bool:
    return mode in {"v2", "dual"}


def should_read_v2(mode: str) -> bool:
    return mode in {"prefer_v2", "v2"}


def should_fallback_v1(mode: str) -> bool:
    return mode in {"prefer_v2", "v1"}


def _safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True)


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_int(value: Any) -> int | None:
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


def source_id(bucket: str, key: str, version_id: str | None = None) -> str:
    normalized = f"s3://{bucket}/{key}?version_id={version_id or ''}"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def split_source_path(source: str | None) -> tuple[str | None, str | None]:
    if not source or "/" not in source:
        return None, None
    bucket, key = source.split("/", 1)
    if not bucket or not key:
        return None, None
    return bucket, key


def extract_text(value: Dict[str, Any]) -> str:
    for key in ("text", "chunk_text", "content", "chunk", "page_content", "raw_text"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def page_range(entry: Dict[str, Any]) -> tuple[int | None, int | None]:
    start = _safe_int(entry.get("page_start"))
    end = _safe_int(entry.get("page_end"))
    pages = entry.get("pages")
    if isinstance(pages, list):
        values = [_safe_int(item) for item in pages]
        numeric = sorted({item for item in values if item is not None})
        if start is None and numeric:
            start = numeric[0]
        if end is None and numeric:
            end = numeric[-1]
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        if start is None:
            start = _safe_int(metadata.get("page_start"))
            if start is None:
                start = _safe_int(metadata.get("page_number"))
        if end is None:
            end = _safe_int(metadata.get("page_end"))
            if end is None:
                end = _safe_int(metadata.get("page_number"))
    if start is None and end is not None:
        start = end
    if end is None and start is not None:
        end = start
    return start, end


def partition_hash(doc_hash: str, entry: Dict[str, Any]) -> str:
    start, end = page_range(entry)
    text = _normalize_spaces(extract_text(entry))
    payload = f"{doc_hash}|{start}|{end}|{text}"
    return _hash_text(payload)


def chunk_hash(doc_hash: str, entry: Dict[str, Any]) -> str:
    start, end = page_range(entry)
    idx = _safe_int(entry.get("chunk_index"))
    text = _normalize_spaces(extract_text(entry))
    payload = f"{doc_hash}|{idx}|{start}|{end}|{text}"
    return _hash_text(payload)


def redis_v2_doc_hashes_key(prefix: str) -> str:
    return f"{prefix}:v2:doc_hashes"


def redis_v2_doc_meta_key(prefix: str, doc_hash: str) -> str:
    return f"{prefix}:v2:doc:{doc_hash}:meta"


def redis_v2_doc_sources_key(prefix: str, doc_hash: str) -> str:
    return f"{prefix}:v2:doc:{doc_hash}:sources"


def redis_v2_doc_collections_key(prefix: str, doc_hash: str) -> str:
    return f"{prefix}:v2:doc:{doc_hash}:collections"


def redis_v2_source_meta_key(prefix: str, sid: str) -> str:
    return f"{prefix}:v2:source:{sid}:meta"


def redis_v2_source_doc_key(prefix: str, sid: str) -> str:
    return f"{prefix}:v2:source:{sid}:doc"


def redis_v2_doc_partition_hashes_key(prefix: str, doc_hash: str) -> str:
    return f"{prefix}:v2:doc:{doc_hash}:partition_hashes"


def redis_v2_partition_key(prefix: str, phash: str) -> str:
    return f"{prefix}:v2:partition:{phash}"


def redis_v2_partition_chunk_hashes_key(prefix: str, phash: str) -> str:
    return f"{prefix}:v2:partition:{phash}:chunk_hashes"


def redis_v2_doc_chunk_hashes_key(prefix: str, doc_hash: str) -> str:
    return f"{prefix}:v2:doc:{doc_hash}:chunk_hashes"


def redis_v2_chunk_key(prefix: str, chash: str) -> str:
    return f"{prefix}:v2:chunk:{chash}"


def bibtex_v2_doc_key(prefix: str, doc_hash: str) -> str:
    return f"{prefix}:v2:doc:{doc_hash}"


def bibtex_v2_source_doc_key(prefix: str, sid: str) -> str:
    return f"{prefix}:v2:source:{sid}:doc"


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _json_load(value: Any, default: Any) -> Any:
    text = _decode(value)
    if not text:
        return default
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default
    return payload


@dataclass(frozen=True)
class SourceDescriptor:
    bucket: str
    key: str
    version_id: str | None = None

    @property
    def source_path(self) -> str:
        return f"{self.bucket}/{self.key}"

    @property
    def source_id(self) -> str:
        return source_id(self.bucket, self.key, self.version_id)


def write_v2_document_payloads(
    redis_client: Any,
    prefix: str,
    doc_hash: str,
    source: SourceDescriptor,
    partitions_payload: List[Dict[str, Any]],
    chunks_payload: List[Dict[str, Any]],
    collection: str | None = None,
) -> Dict[str, str]:
    now = int(__import__("time").time())
    sid = source.source_id

    doc_meta_key = redis_v2_doc_meta_key(prefix, doc_hash)
    doc_sources_key = redis_v2_doc_sources_key(prefix, doc_hash)
    doc_collections_key = redis_v2_doc_collections_key(prefix, doc_hash)
    source_meta_key = redis_v2_source_meta_key(prefix, sid)
    source_doc_key = redis_v2_source_doc_key(prefix, sid)
    doc_partitions_key = redis_v2_doc_partition_hashes_key(prefix, doc_hash)
    doc_chunks_key = redis_v2_doc_chunk_hashes_key(prefix, doc_hash)

    pipe = redis_client.pipeline() if hasattr(redis_client, "pipeline") else None
    writer = pipe if pipe is not None else redis_client

    writer.sadd(redis_v2_doc_hashes_key(prefix), doc_hash)
    writer.sadd(doc_sources_key, sid)
    if collection:
        writer.sadd(doc_collections_key, collection)
    writer.set(source_doc_key, doc_hash)
    writer.hset(
        source_meta_key,
        mapping={
            "bucket": source.bucket,
            "key": source.key,
            "version_id": source.version_id or "",
            "source_path": source.source_path,
            "doc_hash": doc_hash,
            "first_seen_at": str(now),
            "last_seen_at": str(now),
        },
    )
    writer.hset(
        doc_meta_key,
        mapping={
            "doc_hash": doc_hash,
            "partitions_count": str(len(partitions_payload)),
            "chunks_count": str(len(chunks_payload)),
            "primary_source_id": sid,
            "collections_key": doc_collections_key,
            "created_at": str(now),
            "updated_at": str(now),
        },
    )

    writer.delete(doc_partitions_key)
    writer.delete(doc_chunks_key)

    for partition in partitions_payload:
        if not isinstance(partition, dict):
            continue
        phash = partition_hash(doc_hash, partition)
        pkey = redis_v2_partition_key(prefix, phash)
        start, _ = page_range(partition)
        payload = dict(partition)
        payload.setdefault("doc_hash", doc_hash)
        payload.setdefault("partition_hash", phash)
        writer.set(pkey, _safe_json_dumps(payload))
        writer.zadd(doc_partitions_key, {phash: float(start or 0)})

    for chunk in chunks_payload:
        if not isinstance(chunk, dict):
            continue
        chash = chunk_hash(doc_hash, chunk)
        phash = str(chunk.get("partition_hash") or partition_hash(doc_hash, chunk))
        ckey = redis_v2_chunk_key(prefix, chash)
        start, _ = page_range(chunk)
        idx = _safe_int(chunk.get("chunk_index")) or 0
        payload = dict(chunk)
        payload.setdefault("doc_hash", doc_hash)
        payload.setdefault("chunk_hash", chash)
        payload.setdefault("partition_hash", phash)
        payload.setdefault("source_id", sid)
        writer.set(ckey, _safe_json_dumps(payload))
        writer.zadd(doc_chunks_key, {chash: float(idx)})
        writer.zadd(redis_v2_partition_chunk_hashes_key(prefix, phash), {chash: float(idx)})
        if start is not None:
            writer.zadd(doc_chunks_key, {chash: float(idx)})

    if pipe is not None:
        pipe.execute()

    return {
        "doc_meta_key": doc_meta_key,
        "doc_sources_key": doc_sources_key,
        "doc_collections_key": doc_collections_key,
        "source_meta_key": source_meta_key,
        "source_doc_key": source_doc_key,
        "doc_partitions_key": doc_partitions_key,
        "doc_chunks_key": doc_chunks_key,
    }


def read_v2_source_doc_hash(
    redis_client: Any,
    prefix: str,
    bucket: str,
    key: str,
    version_id: str | None = None,
) -> str | None:
    sid = source_id(bucket, key, version_id)
    value = _decode(redis_client.get(redis_v2_source_doc_key(prefix, sid)))
    return str(value) if value else None


def read_v2_doc_chunks(
    redis_client: Any,
    prefix: str,
    doc_hash: str,
) -> List[Dict[str, Any]]:
    index_key = redis_v2_doc_chunk_hashes_key(prefix, doc_hash)
    hashes = []
    if hasattr(redis_client, "zrange"):
        hashes = [_decode(entry) for entry in redis_client.zrange(index_key, 0, -1) or []]
    if not hashes:
        return []
    keys = [redis_v2_chunk_key(prefix, str(entry)) for entry in hashes if entry]
    if not keys:
        return []
    if hasattr(redis_client, "mget"):
        raw_values = redis_client.mget(keys)
    else:
        raw_values = [redis_client.get(key) for key in keys]
    chunks: List[Dict[str, Any]] = []
    for raw in raw_values:
        payload = _json_load(raw, {})
        if isinstance(payload, dict):
            chunks.append(payload)
    return chunks


def read_v2_doc_partitions(
    redis_client: Any,
    prefix: str,
    doc_hash: str,
) -> List[Dict[str, Any]]:
    index_key = redis_v2_doc_partition_hashes_key(prefix, doc_hash)
    hashes = []
    if hasattr(redis_client, "zrange"):
        hashes = [_decode(entry) for entry in redis_client.zrange(index_key, 0, -1) or []]
    if not hashes:
        return []
    keys = [redis_v2_partition_key(prefix, str(entry)) for entry in hashes if entry]
    if not keys:
        return []
    if hasattr(redis_client, "mget"):
        raw_values = redis_client.mget(keys)
    else:
        raw_values = [redis_client.get(key) for key in keys]
    partitions: List[Dict[str, Any]] = []
    for raw in raw_values:
        payload = _json_load(raw, {})
        if isinstance(payload, dict):
            partitions.append(payload)
    return partitions
