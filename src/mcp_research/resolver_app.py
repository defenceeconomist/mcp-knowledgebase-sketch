import json
import os
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from qdrant_client import QdrantClient, models

from mcp_research.link_resolver import resolve_link

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:  # pragma: no cover - optional dependency
    Minio = None
    S3Error = Exception

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None

app = FastAPI(title="Citation Link Resolver", version="0.1.0")


def _decode_redis_value(value):
    """Convert Redis bytes payloads into UTF-8 strings when needed."""
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _load_env_bool(key: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with a default fallback."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_env_list(key: str) -> List[str]:
    """Parse a comma-separated environment variable into a list."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _get_redis_client() -> Tuple[object | None, str | None]:
    """Return a Redis client and optional error message."""
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url or redis is None:
        message = None if redis_url else "REDIS_URL is required for Redis status"
        if redis_url and redis is None:
            message = "redis package is required for Redis status"
        return None, message
    try:
        return redis.from_url(redis_url), None
    except Exception as exc:  # pragma: no cover - runtime connectivity
        return None, f"Failed to connect to Redis: {exc}"


def _get_minio_client() -> Tuple[Any | None, str | None]:
    """Return a MinIO client and optional error message."""
    if Minio is None:
        return None, "minio package is required to list MinIO objects"
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    secure = _load_env_bool("MINIO_SECURE", False)
    if not access_key or not secret_key:
        return None, "MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required to list MinIO objects"
    try:
        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure), None
    except Exception as exc:  # pragma: no cover - runtime connectivity
        return None, f"Failed to connect to MinIO: {exc}"


def _get_qdrant_client() -> QdrantClient | None:
    """Create a Qdrant client using env defaults."""
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
    return QdrantClient(url=qdrant_url)


def _load_partition_count(redis_client, partitions_key: str) -> int:
    """Return the number of partitions stored in Redis for a document."""
    raw = _decode_redis_value(redis_client.get(partitions_key))
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _load_chunk_count(redis_client, chunks_key: str) -> int:
    """Return the number of chunks stored in Redis for a document."""
    raw = _decode_redis_value(redis_client.get(chunks_key))
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _qdrant_uploaded(
    client: QdrantClient | None,
    collection: str,
    document_id: str | None = None,
    bucket: str | None = None,
    key: str | None = None,
    exists_cache: dict | None = None,
    errors: List[str] | None = None,
    error_cache: dict | None = None,
) -> str:
    """Check whether a document appears in a Qdrant collection."""
    if not client:
        return "unknown"
    try:
        cache = exists_cache if exists_cache is not None else {}
        if collection not in cache:
            cache[collection] = client.collection_exists(collection)
        if not cache[collection]:
            return "no"
        must = []
        if document_id:
            must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
        elif bucket and key:
            must.extend(
                [
                    models.FieldCondition(key="bucket", match=models.MatchValue(value=bucket)),
                    models.FieldCondition(key="key", match=models.MatchValue(value=key)),
                ]
            )
        else:
            return "unknown"
        response = client.count(
            collection_name=collection,
            count_filter=models.Filter(must=must),
            exact=True,
        )
    except Exception as exc:
        if errors is not None:
            cache = error_cache if error_cache is not None else {}
            if collection not in cache:
                errors.append(f"Qdrant error for collection {collection}: {exc}")
                cache[collection] = True
        return "unknown"
    return "yes" if response.count > 0 else "no"


def _source_doc_ids(redis_client, redis_prefix: str, source: str) -> List[str]:
    """Look up document ids for a source path in Redis."""
    source_key = f"{redis_prefix}:pdf:source:{source}"
    if hasattr(redis_client, "smembers"):
        raw_members = redis_client.smembers(source_key)
        members = []
        for entry in raw_members or []:
            decoded = _decode_redis_value(entry)
            if decoded:
                members.append(decoded)
        if members:
            return members
    raw = _decode_redis_value(redis_client.get(source_key))
    return [raw] if raw else []


def _resolve_minio_buckets(minio_client: Any | None) -> Tuple[List[str], str | None]:
    """Resolve the list of MinIO buckets to scan from env settings."""
    if not minio_client:
        return [], "MinIO client is not configured"
    buckets = _load_env_list("MINIO_BUCKETS")
    watch_all = _load_env_bool("MINIO_ALL_BUCKETS", False)
    if not buckets and watch_all:
        try:
            buckets = [bucket.name for bucket in minio_client.list_buckets()]
        except S3Error as exc:
            return [], f"Failed to list MinIO buckets: {exc}"
    if not buckets:
        source_bucket = os.getenv("SOURCE_BUCKET", "").strip()
        if source_bucket:
            buckets = [source_bucket]
    if not buckets:
        return [], "No MinIO buckets specified. Set MINIO_BUCKETS or MINIO_ALL_BUCKETS."
    return buckets, None


def _list_minio_objects(minio_client: Any, bucket: str, prefix: str, suffix: str) -> List[str]:
    """List object names in a bucket filtered by prefix/suffix."""
    object_names = []
    for entry in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
        if getattr(entry, "is_dir", False):
            continue
        name = entry.object_name
        if suffix and not name.lower().endswith(suffix.lower()):
            continue
        object_names.append(name)
    return object_names


def _build_minio_index(minio_client: Any | None, errors: List[str]) -> Dict[str, set]:
    """Build an in-memory index of MinIO objects by bucket."""
    if not minio_client:
        return {}
    minio_prefix = os.getenv("MINIO_PREFIX", "").strip()
    minio_suffix = os.getenv("MINIO_SUFFIX", ".pdf").strip()
    index: Dict[str, set] = {}
    bucket_names, bucket_error = _resolve_minio_buckets(minio_client)
    if bucket_error:
        errors.append(bucket_error)
        return index
    for bucket in bucket_names:
        try:
            object_names = _list_minio_objects(minio_client, bucket, minio_prefix, minio_suffix)
        except S3Error as exc:
            errors.append(f"Failed to list objects in {bucket}: {exc}")
            continue
        index[bucket] = set(object_names)
    return index


def _split_source(source: str) -> Tuple[str | None, str | None]:
    """Split a source string into bucket and key components."""
    if not source or "/" not in source:
        return None, None
    bucket, key = source.split("/", 1)
    if not bucket or not key:
        return None, None
    return bucket, key


def _minio_has_source(minio_index: Dict[str, set], source: str, minio_client: Any | None) -> str:
    """Check whether a MinIO index contains a source object."""
    if not minio_client:
        return "unknown"
    bucket, key = _split_source(source)
    if not bucket or not key:
        return "unknown"
    if bucket not in minio_index:
        return "no"
    return "yes" if key in minio_index[bucket] else "no"


def _find_missing_minio_objects() -> Tuple[List[Tuple[str, str]], List[str]]:
    """Find MinIO objects that are missing Redis ingestion records."""
    errors: List[str] = []
    minio_client, minio_error = _get_minio_client()
    if minio_error:
        errors.append(minio_error)
    redis_client, redis_error = _get_redis_client()
    if redis_error:
        errors.append(redis_error)
    if not minio_client or not redis_client:
        return [], errors
    redis_prefix = os.getenv("REDIS_PREFIX", "unstructured")
    minio_prefix = os.getenv("MINIO_PREFIX", "").strip()
    minio_suffix = os.getenv("MINIO_SUFFIX", ".pdf").strip()
    missing: List[Tuple[str, str]] = []
    bucket_names, bucket_error = _resolve_minio_buckets(minio_client)
    if bucket_error:
        errors.append(bucket_error)
        return missing, errors
    for bucket in bucket_names:
        try:
            object_names = _list_minio_objects(minio_client, bucket, minio_prefix, minio_suffix)
        except S3Error as exc:
            errors.append(f"Failed to list objects in {bucket}: {exc}")
            continue
        for object_name in object_names:
            source = f"{bucket}/{object_name}"
            if not _source_doc_ids(redis_client, redis_prefix, source):
                missing.append((bucket, object_name))
    return missing, errors


def _enqueue_missing_ingests(missing: List[Tuple[str, str]]) -> int:
    """Enqueue Celery tasks to ingest missing MinIO objects."""
    if not missing:
        return 0
    from mcp_research.celery_app import celery_app

    for bucket, object_name in missing:
        celery_app.send_task(
            "mcp_research.ingest_minio_object",
            args=[bucket, object_name, None],
        )
    return len(missing)


def _redis_has_doc(redis_client, redis_prefix: str, doc_id: str | None, source: str | None) -> str:
    """Check whether Redis contains metadata for a document or source."""
    if not redis_client:
        return "unknown"
    if doc_id:
        meta_key = f"{redis_prefix}:pdf:{doc_id}:meta"
        if redis_client.exists(meta_key):
            return "yes"
    if source:
        source_key = f"{redis_prefix}:pdf:source:{source}"
        if hasattr(redis_client, "sismember"):
            if doc_id:
                return "yes" if redis_client.sismember(source_key, doc_id) else "no"
            return "yes" if redis_client.scard(source_key) > 0 else "no"
        if redis_client.get(source_key):
            return "yes"
    return "no"


def _build_redis_source_index(redis_client, redis_prefix: str) -> Dict[str, str]:
    """Build a document_id -> source index from Redis metadata."""
    index: Dict[str, str] = {}
    if not redis_client:
        return index
    raw_doc_ids = redis_client.smembers(f"{redis_prefix}:pdf:hashes") or []
    doc_ids = filter(None, (_decode_redis_value(val) for val in raw_doc_ids))
    for doc_id in doc_ids:
        meta_key = f"{redis_prefix}:pdf:{doc_id}:meta"
        meta_raw = redis_client.hgetall(meta_key)
        meta = {_decode_redis_value(k): _decode_redis_value(v) for k, v in meta_raw.items()}
        source = meta.get("source")
        if source:
            index[doc_id] = source
    return index


def _load_qdrant_inventory() -> Tuple[List[dict], List[str]]:
    """Load an inventory of Qdrant points with Redis/MinIO status."""
    errors: List[str] = []
    minio_client, minio_error = _get_minio_client()
    if minio_error:
        errors.append(minio_error)
    redis_client, redis_error = _get_redis_client()
    if redis_error:
        errors.append(redis_error)
    qdrant_client = _get_qdrant_client()
    redis_prefix = os.getenv("REDIS_PREFIX", "unstructured")
    entries: List[dict] = []
    minio_index = _build_minio_index(minio_client, errors)
    redis_source_index = _build_redis_source_index(redis_client, redis_prefix)
    if not qdrant_client:
        return entries, errors
    try:
        collection_list = qdrant_client.get_collections().collections
    except Exception as exc:  # pragma: no cover - runtime connectivity
        errors.append(f"Failed to list Qdrant collections: {exc}")
        return entries, errors
    for collection in collection_list:
        collection_name = collection.name
        offset = None
        grouped: Dict[Tuple[str, str], dict] = {}
        while True:
            try:
                points, next_offset = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:  # pragma: no cover - runtime connectivity
                errors.append(f"Failed to scroll {collection_name}: {exc}")
                break
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("document_id")
                source = payload.get("source")
                if not source:
                    bucket = payload.get("bucket")
                    key = payload.get("key")
                    if bucket and key:
                        source = f"{bucket}/{key}"
                if not source and doc_id:
                    source = redis_source_index.get(doc_id)
                source_display = source or "unknown source"
                group_key = (collection_name, doc_id or source_display)
                grouped.setdefault(
                    group_key,
                    {
                        "collection": collection_name,
                        "doc_id": doc_id or "unknown",
                        "source": source_display,
                        "chunks": 0,
                    },
                )
                grouped[group_key]["chunks"] += 1
            if next_offset is None:
                break
            offset = next_offset
        for entry in grouped.values():
            entry["redis_status"] = _redis_has_doc(
                redis_client,
                redis_prefix,
                None if entry["doc_id"] == "unknown" else entry["doc_id"],
                None if entry["source"] == "unknown source" else entry["source"],
            )
            entry["minio_status"] = _minio_has_source(
                minio_index,
                "" if entry["source"] == "unknown source" else entry["source"],
                minio_client,
            )
            entries.append(entry)
    entries.sort(key=lambda item: (item["collection"], item["source"]))
    return entries, errors


def _render_qdrant_dashboard(entries: List[dict], errors: List[str]) -> str:
    """Render the HTML dashboard for Qdrant inventory status."""
    body = []
    for error in errors:
        body.append(f"<div class='notice'>{escape(error)}</div>")
    if not entries:
        body.append("<div class='empty'>No Qdrant documents found.</div>")
    else:
        body.append("<section class='bucket'>")
        body.append("<table><thead><tr>"
                    "<th>Collection</th><th>Source</th><th>Doc ID</th>"
                    "<th>Chunks</th><th>MinIO</th><th>Redis</th></tr></thead><tbody>")
        for entry in entries:
            minio_label = entry["minio_status"]
            redis_label = entry["redis_status"]
            body.append(
                "<tr>"
                f"<td><div class='meta'>{escape(entry['collection'])}</div></td>"
                f"<td><div class='file'>{escape(entry['source'])}</div></td>"
                f"<td><div class='meta'>{escape(entry['doc_id'])}</div></td>"
                f"<td>{entry['chunks']}</td>"
                f"<td><span class='status {minio_label}'>{minio_label}</span></td>"
                f"<td><span class='status {redis_label}'>{redis_label}</span></td>"
                "</tr>"
            )
        body.append("</tbody></table></section>")
    content = "\n".join(body)
    return _render_page(
        title="Qdrant Inventory",
        subtitle="Files indexed in Qdrant collections with Redis and MinIO presence checks.",
        content=content,
        active_tab="qdrant",
    )


def _load_redis_inventory() -> Tuple[List[dict], List[str]]:
    """Load Redis inventory with MinIO and Qdrant status checks."""
    errors: List[str] = []
    minio_client, minio_error = _get_minio_client()
    if minio_error:
        errors.append(minio_error)
    redis_client, redis_error = _get_redis_client()
    if redis_error:
        errors.append(redis_error)
    qdrant_client = _get_qdrant_client()
    exists_cache: dict = {}
    qdrant_error_cache: dict = {}
    redis_prefix = os.getenv("REDIS_PREFIX", "unstructured")
    entries: List[dict] = []
    if not redis_client:
        return entries, errors
    minio_index = _build_minio_index(minio_client, errors)
    raw_doc_ids = redis_client.smembers(f"{redis_prefix}:pdf:hashes") or []
    doc_ids = sorted(filter(None, (_decode_redis_value(val) for val in raw_doc_ids)))
    for doc_id in doc_ids:
        meta_key = f"{redis_prefix}:pdf:{doc_id}:meta"
        meta_raw = redis_client.hgetall(meta_key)
        meta = {_decode_redis_value(k): _decode_redis_value(v) for k, v in meta_raw.items()}
        source = meta.get("source") or ""
        collections_key = meta.get("collections_key") or f"{redis_prefix}:pdf:{doc_id}:collections"
        collections_raw = redis_client.smembers(collections_key) or []
        collections = sorted(filter(None, (_decode_redis_value(val) for val in collections_raw)))
        bucket, _ = _split_source(source)
        collection_candidates = collections or ([bucket] if bucket else [])
        qdrant_status = "unknown"
        if collection_candidates:
            statuses = [
                _qdrant_uploaded(
                    qdrant_client,
                    collection,
                    document_id=doc_id,
                    exists_cache=exists_cache,
                    errors=errors,
                    error_cache=qdrant_error_cache,
                )
                for collection in collection_candidates
            ]
            if "yes" in statuses:
                qdrant_status = "yes"
            elif all(status == "no" for status in statuses):
                qdrant_status = "no"
        entries.append(
            {
                "doc_id": doc_id,
                "source": source or "unknown source",
                "collections": collections,
                "minio_status": _minio_has_source(minio_index, source, minio_client),
                "qdrant_status": qdrant_status,
            }
        )
    return entries, errors


def _render_redis_dashboard(entries: List[dict], errors: List[str]) -> str:
    """Render the HTML dashboard for Redis inventory."""
    body = []
    for error in errors:
        body.append(f"<div class='notice'>{escape(error)}</div>")
    if not entries:
        body.append("<div class='empty'>No Redis documents found.</div>")
    else:
        body.append("<section class='bucket'>")
        body.append("<table><thead><tr>"
                    "<th>Source</th><th>Doc ID</th><th>Collections</th>"
                    "<th>MinIO</th><th>Qdrant</th></tr></thead><tbody>")
        for entry in entries:
            minio_label = entry["minio_status"]
            qdrant_label = entry["qdrant_status"]
            collections_label = ", ".join(entry["collections"]) if entry["collections"] else "none"
            body.append(
                "<tr>"
                f"<td><div class='file'>{escape(entry['source'])}</div></td>"
                f"<td><div class='meta'>{escape(entry['doc_id'])}</div></td>"
                f"<td><div class='meta'>{escape(collections_label)}</div></td>"
                f"<td><span class='status {minio_label}'>{minio_label}</span></td>"
                f"<td><span class='status {qdrant_label}'>{qdrant_label}</span></td>"
                "</tr>"
            )
        body.append("</tbody></table></section>")
    content = "\n".join(body)
    return _render_page(
        title="Redis Inventory",
        subtitle="Files indexed in Redis with MinIO and Qdrant presence checks.",
        content=content,
        active_tab="redis",
    )


def _load_inventory() -> Tuple[Dict[str, List[dict]], List[str]]:
    """Load combined MinIO/Redis/Qdrant inventory data."""
    errors: List[str] = []
    minio_client, minio_error = _get_minio_client()
    if minio_error:
        errors.append(minio_error)
    redis_client, redis_error = _get_redis_client()
    if redis_error:
        errors.append(redis_error)
    qdrant_client = _get_qdrant_client()
    exists_cache: dict = {}
    qdrant_error_cache: dict = {}
    redis_prefix = os.getenv("REDIS_PREFIX", "unstructured")
    minio_prefix = os.getenv("MINIO_PREFIX", "").strip()
    minio_suffix = os.getenv("MINIO_SUFFIX", ".pdf").strip()

    collections_map: Dict[str, List[dict]] = {}
    if minio_client:
        bucket_names, bucket_error = _resolve_minio_buckets(minio_client)
        if bucket_error:
            errors.append(bucket_error)
        for bucket in bucket_names:
            try:
                object_names = _list_minio_objects(minio_client, bucket, minio_prefix, minio_suffix)
            except S3Error as exc:
                errors.append(f"Failed to list objects in {bucket}: {exc}")
                continue
            for object_name in sorted(object_names):
                source = f"{bucket}/{object_name}"
                doc_ids: List[str] = []
                if redis_client:
                    doc_ids = _source_doc_ids(redis_client, redis_prefix, source)
                doc_id = doc_ids[0] if doc_ids else None
                redis_status = "unknown" if not redis_client else ("yes" if doc_ids else "no")

                partitions_count = None
                chunks_count_value = None
                if redis_client and doc_id:
                    meta_key = f"{redis_prefix}:pdf:{doc_id}:meta"
                    meta_raw = redis_client.hgetall(meta_key)
                    meta = { _decode_redis_value(k): _decode_redis_value(v) for k, v in meta_raw.items() }
                    partitions_key = meta.get("partitions_key") or f"{redis_prefix}:pdf:{doc_id}:partitions"
                    chunks_key = meta.get("chunks_key") or f"{redis_prefix}:pdf:{doc_id}:chunks"
                    partitions_count = _load_partition_count(redis_client, partitions_key)
                    chunks_count = meta.get("chunks")
                    chunks_count_value = int(chunks_count) if str(chunks_count).isdigit() else None
                    if chunks_count_value is None:
                        chunks_count_value = _load_chunk_count(redis_client, chunks_key)

                qdrant_status = _qdrant_uploaded(
                    qdrant_client,
                    bucket,
                    document_id=doc_id,
                    bucket=bucket,
                    key=object_name,
                    exists_cache=exists_cache,
                    errors=errors,
                    error_cache=qdrant_error_cache,
                )
                doc_id_display = doc_id or "not indexed"
                if doc_id and len(doc_ids) > 1:
                    doc_id_display = f"{doc_id} (+{len(doc_ids) - 1})"
                collections_map.setdefault(bucket, []).append(
                    {
                        "doc_id": doc_id_display,
                        "key": object_name,
                        "source": source,
                        "partitions": partitions_count,
                        "chunks": chunks_count_value,
                        "collection": bucket,
                        "redis_status": redis_status,
                        "qdrant_status": qdrant_status,
                    }
                )
    return collections_map, errors


def _render_page(title: str, subtitle: str, content: str, active_tab: str, actions_html: str = "") -> str:
    """Render the full HTML page with tabs and action controls."""
    tab_inventory = "active" if active_tab == "inventory" else ""
    tab_redis = "active" if active_tab == "redis" else ""
    tab_qdrant = "active" if active_tab == "qdrant" else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #1f2328;
      --muted: #5f6b73;
      --accent: #1a6c5c;
      --accent-2: #f4b15a;
      --card: rgba(255, 255, 255, 0.82);
      --border: rgba(17, 24, 39, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top, #fdf6ec, #e8f1ef 45%, #d6e7f0 100%);
      min-height: 100vh;
      padding: 32px 18px 64px;
    }}
    header {{
      max-width: 1100px;
      margin: 0 auto 24px;
      padding: 24px 28px;
      background: var(--card);
      border-radius: 20px;
      border: 1px solid var(--border);
      box-shadow: 0 18px 40px rgba(31, 35, 40, 0.08);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 40px);
      letter-spacing: -0.02em;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .tabs {{
      margin-top: 16px;
      display: inline-flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .actions {{
      margin-top: 16px;
    }}
    .button {{
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 8px 16px;
      font-weight: 600;
      font-size: 13px;
      background: var(--accent);
      color: #f7f7f2;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      box-shadow: 0 10px 18px rgba(26, 108, 92, 0.2);
    }}
    .button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 12px 22px rgba(26, 108, 92, 0.25);
    }}
    .button:active {{
      transform: translateY(0);
      box-shadow: 0 8px 16px rgba(26, 108, 92, 0.2);
    }}
    .tab {{
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 999px;
      border: 1px solid transparent;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.55);
      transition: all 0.2s ease;
    }}
    .tab:hover {{
      color: var(--ink);
      border-color: var(--border);
    }}
    .tab.active {{
      color: var(--accent);
      border-color: rgba(26, 108, 92, 0.35);
      background: rgba(26, 108, 92, 0.12);
    }}
    .notice {{
      max-width: 1100px;
      margin: 0 auto 20px;
      padding: 12px 16px;
      border-radius: 12px;
      background: #fff0f0;
      color: #8a2f2f;
      border: 1px solid #f1b3b3;
    }}
    .info {{
      max-width: 1100px;
      margin: 0 auto 20px;
      padding: 12px 16px;
      border-radius: 12px;
      background: rgba(26, 108, 92, 0.12);
      color: #0d594d;
      border: 1px solid rgba(26, 108, 92, 0.35);
    }}
    .empty {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px;
      border-radius: 16px;
      background: var(--card);
      border: 1px dashed var(--border);
      color: var(--muted);
      text-align: center;
    }}
    .bucket {{
      max-width: 1100px;
      margin: 0 auto 28px;
      padding: 18px 20px;
      background: var(--card);
      border-radius: 18px;
      border: 1px solid var(--border);
      box-shadow: 0 12px 28px rgba(31, 35, 40, 0.06);
      animation: fadeIn 0.6s ease both;
    }}
    .bucket h2 {{
      margin: 4px 0 16px;
      font-size: 22px;
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .file {{
      font-weight: 600;
      font-size: 15px;
    }}
    .meta {{
      font-size: 11px;
      color: var(--muted);
      word-break: break-all;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .status.yes {{
      background: rgba(26, 108, 92, 0.15);
      color: #0d594d;
    }}
    .status.no {{
      background: rgba(184, 64, 64, 0.15);
      color: #a43535;
    }}
    .status.unknown {{
      background: rgba(244, 177, 90, 0.2);
      color: #7a4c12;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p>{escape(subtitle)}</p>
    <div class="tabs">
      <a class="tab {tab_inventory}" href="/ui">MinIO Inventory</a>
      <a class="tab {tab_redis}" href="/ui/redis">Redis Inventory</a>
      <a class="tab {tab_qdrant}" href="/ui/qdrant">Qdrant Inventory</a>
    </div>
    {actions_html}
  </header>
  {content}
</body>
</html>"""


def _render_dashboard(buckets: Dict[str, List[dict]], errors: List[str], infos: List[str] | None = None) -> str:
    """Render the MinIO inventory dashboard HTML."""
    body = []
    for info in infos or []:
        body.append(f"<div class='info'>{escape(info)}</div>")
    for error in errors:
        body.append(f"<div class='notice'>{escape(error)}</div>")
    if not buckets:
        body.append("<div class='empty'>No PDFs found in MinIO.</div>")
    for collection_name, entries in buckets.items():
        body.append(f"<section class='bucket'><h2>{escape(collection_name)}</h2>")
        body.append("<table><thead><tr>"
                    "<th>PDF</th><th>Redis</th><th>Partitions</th><th>Chunks</th>"
                    "<th>Qdrant</th></tr></thead><tbody>")
        for entry in sorted(entries, key=lambda item: item["key"]):
            redis_status = entry["redis_status"]
            status = entry["qdrant_status"]
            redis_label = "unknown" if redis_status == "unknown" else ("yes" if redis_status == "yes" else "no")
            status_label = "unknown" if status == "unknown" else ("yes" if status == "yes" else "no")
            partitions_value = entry["partitions"]
            chunks_value = entry["chunks"]
            body.append(
                "<tr>"
                f"<td><div class='file'>{escape(entry['key'])}</div>"
                f"<div class='meta'>{escape(entry['doc_id'])}</div></td>"
                f"<td><span class='status {redis_label}'>{redis_label}</span></td>"
                f"<td>{'-' if partitions_value is None else partitions_value}</td>"
                f"<td>{'-' if chunks_value is None else chunks_value}</td>"
                f"<td><span class='status {status_label}'>{status_label}</span></td>"
                "</tr>"
            )
        body.append("</tbody></table></section>")

    content = "\n".join(body)
    actions_html = (
        "<form class='actions' method='post' action='/ui/ingest-missing'>"
        "<button class='button' type='submit'>Ingest Missing Files</button>"
        "</form>"
    )
    return _render_page(
        title="PDF Inventory",
        subtitle="Grouped by collection with MinIO files plus Redis and Qdrant status.",
        content=content,
        active_tab="inventory",
        actions_html=actions_html,
    )


@app.get("/r/doc")
def resolve_doc(
    ref: Optional[str] = Query(default=None, description="doc:// reference"),
    bucket: Optional[str] = Query(default=None),
    key: Optional[str] = Query(default=None),
    version_id: Optional[str] = Query(default=None),
    page: Optional[int] = Query(default=None),
    page_start: Optional[int] = Query(default=None),
    page_end: Optional[int] = Query(default=None),
    mode: Optional[str] = Query(default=None),
):
    """Resolve a citation reference to a target URL."""
    try:
        result = resolve_link(
            source_ref=ref,
            bucket=bucket,
            key=key,
            version_id=version_id,
            page=page,
            page_start=page_start,
            page_end=page_end,
            mode=mode,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=result["url"], status_code=302)


@app.get("/r/doc.json")
def resolve_doc_json(
    ref: Optional[str] = Query(default=None, description="doc:// reference"),
    bucket: Optional[str] = Query(default=None),
    key: Optional[str] = Query(default=None),
    version_id: Optional[str] = Query(default=None),
    page: Optional[int] = Query(default=None),
    page_start: Optional[int] = Query(default=None),
    page_end: Optional[int] = Query(default=None),
    mode: Optional[str] = Query(default=None),
):
    """Resolve a citation reference and return JSON instead of redirecting."""
    try:
        result = resolve_link(
            source_ref=ref,
            bucket=bucket,
            key=key,
            version_id=version_id,
            page=page,
            page_start=page_start,
            page_end=page_end,
            mode=mode,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=result)


@app.get("/ui")
def inventory_ui(notice: Optional[str] = Query(default=None)):
    """Render the inventory dashboard UI."""
    buckets, errors = _load_inventory()
    infos = [notice] if notice else None
    html = _render_dashboard(buckets, errors, infos=infos)
    return HTMLResponse(content=html)


@app.post("/ui/ingest-missing")
def ingest_missing_ui():
    """Enqueue missing ingests and return the refreshed dashboard."""
    missing, enqueue_errors = _find_missing_minio_objects()
    count = _enqueue_missing_ingests(missing)
    buckets, errors = _load_inventory()
    errors.extend(enqueue_errors)
    infos = [f"Enqueued {count} ingest task(s) for missing MinIO files."]
    html = _render_dashboard(buckets, errors, infos=infos)
    return HTMLResponse(content=html)


@app.get("/ui/redis")
def redis_inventory_ui():
    """Render the Redis inventory UI."""
    entries, errors = _load_redis_inventory()
    html = _render_redis_dashboard(entries, errors)
    return HTMLResponse(content=html)


@app.get("/ui/qdrant")
def qdrant_inventory_ui():
    """Render the Qdrant inventory UI."""
    entries, errors = _load_qdrant_inventory()
    html = _render_qdrant_dashboard(entries, errors)
    return HTMLResponse(content=html)


def main() -> None:
    """CLI entry point for the resolver API server."""
    host = os.getenv("RESOLVER_HOST", "0.0.0.0")
    port = int(os.getenv("RESOLVER_PORT", "8080"))
    import uvicorn

    uvicorn.run("mcp_research.resolver_app:app", host=host, port=port)


if __name__ == "__main__":
    main()
