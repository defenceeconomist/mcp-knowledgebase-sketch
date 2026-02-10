from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

try:
    from qdrant_client import QdrantClient, models
except ImportError:  # pragma: no cover - allow importing helpers without Qdrant
    QdrantClient = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
except ImportError:  # pragma: no cover - allow importing helpers without FastAPI
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

    def Query(default, **_kwargs):  # type: ignore[override]
        return default

    HTMLResponse = str  # type: ignore[assignment]


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dotenv(path: str = ".env") -> None:
    """Load a .env file into the process environment if present."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_URL = os.getenv("QDRANT_URL") or f"http://{QDRANT_HOST}:{QDRANT_PORT}"
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "unstructured")

_qdrant_client: QdrantClient | None = None
_redis_client = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if QdrantClient is None:  # pragma: no cover
            raise RuntimeError("qdrant-client is required to use the dashboard")
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def _get_redis_client():
    global _redis_client
    if not REDIS_URL:
        return None
    if redis is None:
        logger.warning("REDIS_URL set but redis package is missing; skipping Redis")
        return None
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL)
    return _redis_client


def _decode_redis_value(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _redis_key(prefix: str, doc_id: str, suffix: str) -> str:
    return f"{prefix}:pdf:{doc_id}:{suffix}"


def _source_key(prefix: str, source: str) -> str:
    return f"{prefix}:pdf:source:{source}"


def _split_source(source: str | None) -> Tuple[str | None, str | None]:
    if not source or "/" not in source:
        return None, None
    bucket, key = source.split("/", 1)
    if not bucket or not key:
        return None, None
    return bucket, key


def _coerce_qdrant_offset(offset: str | None):
    if offset is None:
        return None
    if offset.isdigit():
        return int(offset)
    return offset


def _file_identity(payload: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    document_id = payload.get("document_id")
    source = payload.get("source")
    bucket = payload.get("bucket")
    key = payload.get("key")
    if document_id:
        identity = f"doc:{document_id}"
    elif source:
        identity = f"source:{source}"
    elif bucket and key:
        identity = f"object:{bucket}/{key}"
    else:
        return None
    return identity, {
        "document_id": document_id,
        "source": source,
        "bucket": bucket,
        "key": key,
    }


def _safe_json_list_len(raw: str | None) -> Optional[int]:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return len(payload) if isinstance(payload, list) else None


def _normalize_metadata_value(value: Any) -> Any:
    """Convert metadata values into JSON-safe, UI-friendly payloads."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value if len(value) <= 500 else f"{value[:497]}..."
    if isinstance(value, list):
        return [_normalize_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _normalize_metadata_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _extract_qdrant_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract representative Qdrant metadata while omitting likely chunk body fields."""
    skip_keys = {"text", "content", "chunk", "chunk_text", "page_content", "raw_text"}
    return {
        key: _normalize_metadata_value(value)
        for key, value in payload.items()
        if key not in skip_keys
    }


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
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


def _first_page_number(payload: Dict[str, Any]) -> int | None:
    page_start = _coerce_int(payload.get("page_start"))
    if page_start is not None:
        return page_start
    pages = payload.get("pages")
    if isinstance(pages, list):
        page_values = [num for num in (_coerce_int(value) for value in pages) if num is not None]
        if page_values:
            return min(page_values)
    page_end = _coerce_int(payload.get("page_end"))
    return page_end


def _chunk_order_key(payload: Dict[str, Any]) -> Tuple[float, float]:
    page = _first_page_number(payload)
    chunk_index = _coerce_int(payload.get("chunk_index"))
    page_rank = float(page) if page is not None else float("inf")
    chunk_rank = float(chunk_index) if chunk_index is not None else float("inf")
    return page_rank, chunk_rank


def _apply_qdrant_exemplar(rec: Dict[str, Any], payload: Dict[str, Any]) -> None:
    rec["qdrant_metadata"] = _extract_qdrant_metadata(payload)
    for key in ("version_id", "page_start", "page_end", "pages", "source_ref"):
        if payload.get(key) is not None:
            rec[key] = payload.get(key)
        else:
            rec.pop(key, None)


def _sorted_pages(payload: Dict[str, Any]) -> List[int]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []
    values = [_coerce_int(value) for value in pages]
    return sorted({value for value in values if value is not None})


def _chunk_partition_info(payload: Dict[str, Any]) -> Tuple[str, int | None, int | None]:
    page_start = _coerce_int(payload.get("page_start"))
    page_end = _coerce_int(payload.get("page_end"))
    pages = _sorted_pages(payload)

    if page_start is None and pages:
        page_start = pages[0]
    if page_end is None and pages:
        page_end = pages[-1]
    if page_start is None and page_end is not None:
        page_start = page_end
    if page_end is None and page_start is not None:
        page_end = page_start

    if page_start is None:
        return "Unknown pages", None, None
    if page_end is None or page_end == page_start:
        return f"Page {page_start}", page_start, page_start
    return f"Pages {page_start}-{page_end}", page_start, page_end


def _chunk_sort_key(payload: Dict[str, Any]) -> Tuple[float, float, float, str]:
    page_rank, chunk_rank = _chunk_order_key(payload)
    page_end = _coerce_int(payload.get("page_end"))
    end_rank = float(page_end) if page_end is not None else page_rank
    source_ref = str(payload.get("source_ref") or "")
    return page_rank, end_rank, chunk_rank, source_ref


def _extract_chunk_text(payload: Dict[str, Any]) -> str:
    for key in ("text", "chunk_text", "content", "chunk", "page_content", "raw_text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _chunk_detail_entry(point_id: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    partition_label, partition_page_start, partition_page_end = _chunk_partition_info(payload)
    entry = {
        "point_id": str(point_id),
        "chunk_index": _coerce_int(payload.get("chunk_index")),
        "page_start": _coerce_int(payload.get("page_start")),
        "page_end": _coerce_int(payload.get("page_end")),
        "pages": _sorted_pages(payload),
        "source_ref": payload.get("source_ref"),
        "partition_label": partition_label,
        "partition_page_start": partition_page_start,
        "partition_page_end": partition_page_end,
        "text": _extract_chunk_text(payload),
    }
    return entry


def _chunk_payload_identity_key(payload: Dict[str, Any], point_id: Any | None = None) -> Tuple[Any, ...]:
    pages = _sorted_pages(payload)
    chunk_index = _coerce_int(payload.get("chunk_index"))
    page_start = _coerce_int(payload.get("page_start"))
    page_end = _coerce_int(payload.get("page_end"))
    source_ref = payload.get("source_ref")
    text = _extract_chunk_text(payload)
    has_payload_identity = any(
        value not in (None, "", ())
        for value in (chunk_index, page_start, page_end, tuple(pages), source_ref, text)
    )
    if not has_payload_identity:
        return ("point", str(point_id))
    return ("payload", chunk_index, page_start, page_end, tuple(pages), source_ref, text)


def group_chunks_by_partition(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []
    for chunk in chunks:
        start = chunk.get("partition_page_start")
        end = chunk.get("partition_page_end")
        label = chunk.get("partition_label") or "Unknown pages"
        key = f"{start}:{end}:{label}"
        if key not in groups:
            groups[key] = {
                "label": label,
                "page_start": start,
                "page_end": end,
                "chunks": [],
            }
            ordered_keys.append(key)
        groups[key]["chunks"].append(chunk)

    partitions: List[Dict[str, Any]] = []
    for idx, key in enumerate(ordered_keys, start=1):
        partition = groups[key]
        partitions.append(
            {
                "partition_index": idx,
                "label": partition["label"],
                "page_start": partition["page_start"],
                "page_end": partition["page_end"],
                "chunk_count": len(partition["chunks"]),
                "chunks": partition["chunks"],
            }
        )
    return partitions


def _chunk_entry_identity_key(chunk: Dict[str, Any]) -> Tuple[Any, ...]:
    pages = chunk.get("pages") or []
    if not isinstance(pages, list):
        pages = []
    return (
        chunk.get("chunk_index"),
        chunk.get("page_start"),
        chunk.get("page_end"),
        tuple(pages),
        chunk.get("source_ref"),
        chunk.get("text"),
    )


def dedupe_chunk_entries(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_payloads: set[Tuple[Any, ...]] = set()
    for chunk in chunks:
        point_id = str(chunk.get("point_id") or "")
        if point_id and point_id in seen_ids:
            continue
        payload_key = _chunk_entry_identity_key(chunk)
        if payload_key in seen_payloads:
            continue
        if point_id:
            seen_ids.add(point_id)
        seen_payloads.add(payload_key)
        deduped.append(chunk)
    return deduped


def _build_qdrant_chunk_filter(
    collection_name: str,
    document_id: str | None,
    source: str | None,
    key: str | None,
) -> Any:
    if models is None:  # pragma: no cover - runtime dependency
        raise RuntimeError("qdrant-client models are required for chunk filtering")

    must: List[Any] = []
    if document_id:
        must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
    elif source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    elif key:
        must.extend(
            [
                models.FieldCondition(key="bucket", match=models.MatchValue(value=collection_name)),
                models.FieldCondition(key="key", match=models.MatchValue(value=key)),
            ]
        )
    else:
        raise ValueError("document_id, source, or key is required")
    return models.Filter(must=must)


def fetch_file_chunks(
    client: QdrantClient,
    collection_name: str,
    document_id: str | None,
    source: str | None,
    key: str | None,
    batch_size: int,
    limit: int,
) -> Dict[str, Any]:
    scroll_filter = _build_qdrant_chunk_filter(collection_name, document_id, source, key)
    batch_size = max(1, batch_size)
    limit = max(1, limit)

    chunks: List[Dict[str, Any]] = []
    scanned_points = 0
    next_offset = None
    truncated = False

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )
        if not points:
            break
        scanned_points += len(points)
        for point in points:
            payload = point.payload or {}
            chunks.append(_chunk_detail_entry(point.id, payload))
            if len(chunks) >= limit:
                truncated = True
                break
        if truncated or next_offset is None:
            break

    chunks.sort(key=_chunk_sort_key)
    raw_count = len(chunks)
    chunks = dedupe_chunk_entries(chunks)
    partitions = group_chunks_by_partition(chunks)
    return {
        "collection": collection_name,
        "count": len(chunks),
        "raw_count": raw_count,
        "partition_count": len(partitions),
        "scanned_points": scanned_points,
        "truncated": truncated,
        "chunks": chunks,
        "partitions": partitions,
    }


def _partition_summaries(partitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "partition_index": partition["partition_index"],
            "label": partition["label"],
            "page_start": partition["page_start"],
            "page_end": partition["page_end"],
            "chunk_count": partition["chunk_count"],
        }
        for partition in partitions
    ]


def fetch_file_partition_summaries(
    client: QdrantClient,
    collection_name: str,
    document_id: str | None,
    source: str | None,
    key: str | None,
    batch_size: int,
    limit: int,
) -> Dict[str, Any]:
    details = fetch_file_chunks(
        client=client,
        collection_name=collection_name,
        document_id=document_id,
        source=source,
        key=key,
        batch_size=batch_size,
        limit=limit,
    )
    partitions = details.get("partitions") or []
    return {
        "collection": details.get("collection"),
        "count": details.get("count"),
        "raw_count": details.get("raw_count"),
        "partition_count": details.get("partition_count"),
        "scanned_points": details.get("scanned_points"),
        "truncated": details.get("truncated"),
        "partitions": _partition_summaries(partitions),
    }


def fetch_partition_chunks(
    client: QdrantClient,
    collection_name: str,
    document_id: str | None,
    source: str | None,
    key: str | None,
    partition_index: int,
    batch_size: int,
    limit: int,
) -> Dict[str, Any]:
    if partition_index < 1:
        raise ValueError("partition_index must be >= 1")
    details = fetch_file_chunks(
        client=client,
        collection_name=collection_name,
        document_id=document_id,
        source=source,
        key=key,
        batch_size=batch_size,
        limit=limit,
    )
    partitions = details.get("partitions") or []
    selected = next(
        (partition for partition in partitions if int(partition.get("partition_index") or -1) == partition_index),
        None,
    )
    if selected is None:
        raise ValueError(f"Partition not found: {partition_index}")
    return {
        "collection": details.get("collection"),
        "count": details.get("count"),
        "raw_count": details.get("raw_count"),
        "partition_count": details.get("partition_count"),
        "scanned_points": details.get("scanned_points"),
        "truncated": details.get("truncated"),
        "partition": {
            "partition_index": selected.get("partition_index"),
            "label": selected.get("label"),
            "page_start": selected.get("page_start"),
            "page_end": selected.get("page_end"),
            "chunk_count": selected.get("chunk_count"),
            "chunks": selected.get("chunks") or [],
        },
    }


def _build_source_ref(bucket: str, key: str, version_id: str | None = None) -> str:
    safe_key = quote(key.lstrip("/"), safe="/")
    source_ref = f"doc://{bucket}/{safe_key}"
    if version_id:
        source_ref = f"{source_ref}?version_id={quote(version_id, safe='')}"
    return source_ref


def _build_original_file_url(entry: Dict[str, Any]) -> Optional[str]:
    source_ref = entry.get("source_ref")
    bucket = entry.get("bucket")
    key = entry.get("key")
    source = entry.get("source")
    if not source_ref:
        if not (bucket and key):
            source_bucket, source_key = _split_source(source)
            bucket = bucket or source_bucket
            key = key or source_key
        if not (bucket and key):
            return None
        source_ref = _build_source_ref(bucket, key, entry.get("version_id"))

    base = (
        os.getenv("CITATION_BASE_URL")
        or os.getenv("DOCS_BASE_URL")
        or os.getenv("LINK_RESOLVER_BASE_URL")
        or "http://localhost:8080"
    ).rstrip("/")
    ref_path = os.getenv("CITATION_REF_PATH", "/r/doc")
    if not ref_path.startswith("/"):
        ref_path = f"/{ref_path}"
    return f"{base}{ref_path}?ref={quote(source_ref, safe='')}"


def attach_original_file_links(files: List[Dict[str, Any]]) -> None:
    for entry in files:
        entry["original_file_url"] = _build_original_file_url(entry)


def scan_collection_files(
    client: QdrantClient,
    collection_name: str,
    limit: int,
    batch_size: int,
    offset: str | None,
) -> Dict[str, Any]:
    limit = max(1, limit)
    batch_size = max(1, batch_size)

    files: Dict[str, Dict[str, Any]] = {}
    scanned_points = 0
    next_offset = _coerce_qdrant_offset(offset)
    last_point_id = None
    truncated = False

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

        scanned_points += len(points)
        for point in points:
            payload = point.payload or {}
            entry = _file_identity(payload)
            if not entry:
                continue
            last_point_id = point.id
            identity, info = entry
            rec = files.get(identity)
            if rec is None:
                rec = dict(info)
                if not rec.get("bucket"):
                    rec["bucket"] = collection_name
                rec["qdrant_chunks"] = 0
                rec["qdrant_partitions"] = 0
                rec["__qdrant_first_chunk_rank"] = _chunk_order_key(payload)
                rec["__qdrant_chunk_keys"] = set()
                rec["__qdrant_partition_keys"] = set()
                _apply_qdrant_exemplar(rec, payload)
                files[identity] = rec
            else:
                rank = _chunk_order_key(payload)
                if rank < rec.get("__qdrant_first_chunk_rank", (float("inf"), float("inf"))):
                    rec["__qdrant_first_chunk_rank"] = rank
                    _apply_qdrant_exemplar(rec, payload)
            chunk_key = _chunk_payload_identity_key(payload, point.id)
            if chunk_key not in rec["__qdrant_chunk_keys"]:
                rec["__qdrant_chunk_keys"].add(chunk_key)
                rec["qdrant_chunks"] += 1
                partition_label, partition_page_start, partition_page_end = _chunk_partition_info(payload)
                partition_key = (partition_label, partition_page_start, partition_page_end)
                rec["__qdrant_partition_keys"].add(partition_key)
                rec["qdrant_partitions"] = len(rec["__qdrant_partition_keys"])
            if len(files) >= limit:
                truncated = True
                if last_point_id is not None:
                    next_offset = last_point_id
                break

        if truncated or next_offset is None:
            break

    return {
        "collection": collection_name,
        "count": len(files),
        "scanned_points": scanned_points,
        "next_offset": str(next_offset) if next_offset is not None else None,
        "truncated": truncated,
        "files": [{k: v for k, v in item.items() if not k.startswith("__")} for item in files.values()],
    }


def _source_doc_ids(redis_client, prefix: str, source: str) -> List[str]:
    source_key = _source_key(prefix, source)
    doc_ids: List[str] = []
    if hasattr(redis_client, "smembers"):
        raw_members = redis_client.smembers(source_key)
        for entry in raw_members or []:
            decoded = _decode_redis_value(entry)
            if decoded:
                doc_ids.append(decoded)
        if doc_ids:
            return doc_ids
    raw = _decode_redis_value(redis_client.get(source_key))
    return [raw] if raw else []


def enrich_files_with_redis(
    redis_client,
    prefix: str,
    collection_name: str,
    files: List[Dict[str, Any]],
) -> None:
    if not redis_client or not files:
        return

    primary_doc_ids: List[str | None] = []
    for entry in files:
        doc_id = entry.get("document_id")
        bucket = entry.get("bucket") or collection_name
        key = entry.get("key")
        source = entry.get("source") or (f"{bucket}/{key}" if key else None)
        if doc_id:
            primary_doc_ids.append(doc_id)
            entry["redis_doc_ids"] = [doc_id]
            continue
        if source:
            doc_ids = _source_doc_ids(redis_client, prefix, source)
            entry["redis_doc_ids"] = doc_ids
            primary_doc_ids.append(doc_ids[0] if doc_ids else None)
        else:
            entry["redis_doc_ids"] = []
            primary_doc_ids.append(None)

    idx_with_doc: List[int] = [idx for idx, doc_id in enumerate(primary_doc_ids) if doc_id]
    raw_metas: List[Any] = [None] * len(primary_doc_ids)
    if idx_with_doc:
        pipe = redis_client.pipeline()
        for idx in idx_with_doc:
            pipe.hgetall(_redis_key(prefix, primary_doc_ids[idx], "meta"))
        results = pipe.execute()
        for idx, raw_meta in zip(idx_with_doc, results):
            raw_metas[idx] = raw_meta

    partition_keys: List[str | None] = []
    decoded_metas: List[Dict[str, Any] | None] = []
    for doc_id, raw_meta in zip(primary_doc_ids, raw_metas):
        if not doc_id:
            decoded_metas.append(None)
            partition_keys.append(None)
            continue
        meta: Dict[str, Any] = {}
        if isinstance(raw_meta, dict):
            for k, v in raw_meta.items():
                meta[_decode_redis_value(k)] = _decode_redis_value(v)
        decoded_metas.append(meta)
        partition_keys.append(meta.get("partitions_key") or _redis_key(prefix, doc_id, "partitions"))

    idx_with_partitions: List[int] = [idx for idx, pkey in enumerate(partition_keys) if pkey]
    raw_partitions: List[Any] = [None] * len(partition_keys)
    if idx_with_partitions:
        pipe = redis_client.pipeline()
        for idx in idx_with_partitions:
            pipe.get(partition_keys[idx])
        results = pipe.execute()
        for idx, raw in zip(idx_with_partitions, results):
            raw_partitions[idx] = raw

    for entry, meta, partitions_raw in zip(files, decoded_metas, raw_partitions):
        partitions_decoded = _decode_redis_value(partitions_raw)
        redis_partitions = _safe_json_list_len(partitions_decoded)

        redis_chunks = None
        if meta and meta.get("chunks") is not None:
            try:
                redis_chunks = int(str(meta.get("chunks")))
            except ValueError:
                redis_chunks = None

        entry["redis_chunks"] = redis_chunks
        entry["redis_partitions"] = redis_partitions
        entry["redis_meta_key"] = (
            _redis_key(prefix, entry["redis_doc_ids"][0], "meta") if entry.get("redis_doc_ids") else None
        )
        entry["redis_metadata"] = {k: _normalize_metadata_value(v) for k, v in (meta or {}).items()} if meta else None


def _load_dashboard_html() -> str:
    try:
        from importlib.resources import files as pkg_files  # Python 3.9+

        return pkg_files("mcp_research.dashboard_static").joinpath("index.html").read_text(
            encoding="utf-8"
        )
    except Exception:  # pragma: no cover
        return "<html><body><h1>Dashboard assets missing</h1></body></html>"


app = FastAPI(title="MCP Research Dashboard") if FastAPI is not None else None


if app is not None:

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_load_dashboard_html())

    @app.get("/api/buckets")
    def api_buckets() -> Dict[str, Any]:
        client = _get_qdrant_client()
        collections = client.get_collections()
        buckets = sorted([entry.name for entry in collections.collections])
        return {"buckets": buckets}

    @app.get("/api/buckets/{bucket}/files")
    def api_bucket_files(
        bucket: str,
        limit: int = Query(200, ge=1, le=2000),
        batch_size: int = Query(256, ge=1, le=2048),
        offset: str | None = None,
    ) -> Dict[str, Any]:
        client = _get_qdrant_client()
        if not client.collection_exists(bucket):
            raise HTTPException(status_code=404, detail=f"Qdrant collection not found: {bucket}")

        result = scan_collection_files(
            client=client,
            collection_name=bucket,
            limit=limit,
            batch_size=batch_size,
            offset=offset,
        )
        files = result.get("files") or []

        redis_client = _get_redis_client()
        if redis_client:
            enrich_files_with_redis(redis_client, REDIS_PREFIX, bucket, files)
        attach_original_file_links(files)
        result["files"] = files
        result["bucket"] = bucket
        return result

    @app.get("/api/buckets/{bucket}/files/chunks")
    def api_bucket_file_chunks(
        bucket: str,
        document_id: str | None = None,
        source: str | None = None,
        key: str | None = None,
        batch_size: int = Query(256, ge=1, le=2048),
        limit: int = Query(5000, ge=1, le=50000),
    ) -> Dict[str, Any]:
        client = _get_qdrant_client()
        if not client.collection_exists(bucket):
            raise HTTPException(status_code=404, detail=f"Qdrant collection not found: {bucket}")
        try:
            result = fetch_file_chunks(
                client=client,
                collection_name=bucket,
                document_id=document_id,
                source=source,
                key=key,
                batch_size=batch_size,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result["bucket"] = bucket
        return result

    @app.get("/api/buckets/{bucket}/files/partitions")
    def api_bucket_file_partitions(
        bucket: str,
        document_id: str | None = None,
        source: str | None = None,
        key: str | None = None,
        batch_size: int = Query(256, ge=1, le=2048),
        limit: int = Query(5000, ge=1, le=50000),
    ) -> Dict[str, Any]:
        client = _get_qdrant_client()
        if not client.collection_exists(bucket):
            raise HTTPException(status_code=404, detail=f"Qdrant collection not found: {bucket}")
        try:
            result = fetch_file_partition_summaries(
                client=client,
                collection_name=bucket,
                document_id=document_id,
                source=source,
                key=key,
                batch_size=batch_size,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result["bucket"] = bucket
        return result

    @app.get("/api/buckets/{bucket}/files/partitions/{partition_index}/chunks")
    def api_bucket_partition_chunks(
        bucket: str,
        partition_index: int,
        document_id: str | None = None,
        source: str | None = None,
        key: str | None = None,
        batch_size: int = Query(256, ge=1, le=2048),
        limit: int = Query(5000, ge=1, le=50000),
    ) -> Dict[str, Any]:
        client = _get_qdrant_client()
        if not client.collection_exists(bucket):
            raise HTTPException(status_code=404, detail=f"Qdrant collection not found: {bucket}")
        try:
            result = fetch_partition_chunks(
                client=client,
                collection_name=bucket,
                document_id=document_id,
                source=source,
                key=key,
                partition_index=partition_index,
                batch_size=batch_size,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result["bucket"] = bucket
        return result


def main() -> None:
    import uvicorn  # pylint: disable=import-outside-toplevel

    if app is None:  # pragma: no cover
        raise RuntimeError("fastapi is required to run the dashboard")

    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8002"))
    uvicorn.run("mcp_research.dashboard_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
