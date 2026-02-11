from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote, urlencode

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
except ImportError:  # pragma: no cover - keep importable without FastAPI
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

    def Query(default, **_kwargs):  # type: ignore[override]
        return default

    HTMLResponse = str  # type: ignore[assignment]

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:  # pragma: no cover - optional dependency
    Minio = None  # type: ignore[assignment]
    S3Error = Exception  # type: ignore[assignment]

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore[assignment]

try:
    from mcp_research import bibtex_autofill
except Exception:  # pragma: no cover - optional dependency chain
    bibtex_autofill = None  # type: ignore[assignment]

try:
    from mcp_research.link_resolver import build_citation_url, build_source_ref
except Exception:  # pragma: no cover - fallback when optional deps are unavailable

    def build_source_ref(
        bucket: str,
        key: str,
        page_start: int | None = None,
        page_end: int | None = None,
        version_id: str | None = None,
    ) -> str:
        if not bucket:
            raise ValueError("bucket is required to build source_ref")
        if not key:
            raise ValueError("key is required to build source_ref")
        safe_key = quote(key.lstrip("/"), safe="/")
        fragment = ""
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                fragment = f"page={page_start}-{page_end}"
            else:
                fragment = f"page={page_start}"
        query = urlencode({"version_id": version_id}) if version_id else ""
        ref = f"doc://{bucket}/{safe_key}"
        if query:
            ref = f"{ref}?{query}"
        if fragment:
            ref = f"{ref}#{fragment}"
        return ref

    def build_citation_url(
        source_ref: str,
        base_url: str | None = None,
        ref_path: str | None = None,
    ) -> str | None:
        base = (
            base_url
            or os.getenv("CITATION_BASE_URL")
            or os.getenv("DOCS_BASE_URL")
            or "http://localhost:8080"
        )
        path = ref_path or os.getenv("CITATION_REF_PATH", "/r/doc")
        base = base.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        encoded_ref = quote(source_ref, safe="")
        return f"{base}{path}?ref={encoded_ref}"


ALLOWED_ENTRY_TYPES = {"article", "inproceedings", "book", "misc", "techreport"}


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


def _load_ui_html() -> str:
    try:
        from importlib.resources import files as pkg_files  # Python 3.9+

        return pkg_files("mcp_research.bibtex_ui_static").joinpath("index.html").read_text(
            encoding="utf-8"
        )
    except Exception:  # pragma: no cover
        return "<html><body><h1>BibTeX UI assets missing</h1></body></html>"


def _decode_redis_value(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _load_env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_env_list(key: str) -> List[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _get_redis_client() -> Tuple[Any | None, str | None]:
    redis_url = os.getenv("BIBTEX_REDIS_URL") or os.getenv("REDIS_URL", "")
    if not redis_url:
        return None, "REDIS_URL (or BIBTEX_REDIS_URL) is required"
    if redis is None:
        return None, "redis package is required for BibTeX storage"
    try:
        return redis.from_url(redis_url), None
    except Exception as exc:  # pragma: no cover - runtime connectivity
        return None, f"Failed to connect to Redis: {exc}"


def _get_minio_client() -> Tuple[Any | None, str | None]:
    if Minio is None:
        return None, "minio package is required to browse buckets"
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    secure = _load_env_bool("MINIO_SECURE", False)
    if not access_key or not secret_key:
        return None, "MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required"
    try:
        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure), None
    except Exception as exc:  # pragma: no cover - runtime connectivity
        return None, f"Failed to connect to MinIO: {exc}"


def _resolve_minio_buckets(minio_client: Any | None) -> Tuple[List[str], str | None]:
    if not minio_client:
        return [], "MinIO client is not configured"

    buckets = _load_env_list("BIBTEX_MINIO_BUCKETS")
    if not buckets:
        buckets = _load_env_list("MINIO_BUCKETS")

    watch_all = _load_env_bool("BIBTEX_MINIO_ALL_BUCKETS", _load_env_bool("MINIO_ALL_BUCKETS", False))
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
        return [], "No MinIO buckets configured. Set BIBTEX_MINIO_BUCKETS or BIBTEX_MINIO_ALL_BUCKETS=1."

    return sorted(set(buckets)), None


def _list_minio_objects(
    minio_client: Any,
    bucket: str,
    prefix: str,
    suffix: str,
    limit: int,
) -> List[str]:
    object_names: List[str] = []
    for entry in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
        if getattr(entry, "is_dir", False):
            continue
        object_name = entry.object_name
        if suffix and not object_name.lower().endswith(suffix.lower()):
            continue
        object_names.append(object_name)
        if len(object_names) >= limit:
            break
    return sorted(object_names)


def _bibtex_prefix() -> str:
    raw = os.getenv("BIBTEX_REDIS_PREFIX", "bibtex").strip()
    return raw or "bibtex"


def _bibtex_file_key(prefix: str, bucket: str, object_name: str) -> str:
    return f"{prefix}:file:{bucket}/{object_name}"


def _source_redis_prefix() -> str:
    raw = os.getenv("BIBTEX_SOURCE_REDIS_PREFIX", os.getenv("REDIS_PREFIX", "unstructured")).strip()
    return raw or "unstructured"


def _source_key(prefix: str, source: str) -> str:
    return f"{prefix}:pdf:source:{source}"


def _source_doc_ids(redis_client: Any, prefix: str, source: str) -> List[str]:
    source_key = _source_key(prefix, source)
    values: List[str] = []
    if hasattr(redis_client, "smembers"):
        raw_members = redis_client.smembers(source_key)
        for entry in raw_members or []:
            decoded = _decode_redis_value(entry)
            if decoded:
                values.append(str(decoded))
    if not values:
        raw = _decode_redis_value(redis_client.get(source_key))
        if raw:
            values.append(str(raw))
    return sorted({value for value in values if value})


def _safe_json_list(raw_value: Any) -> List[Any]:
    raw = _decode_redis_value(raw_value)
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return payload if isinstance(payload, list) else []


def _doc_meta(redis_client: Any, prefix: str, doc_id: str) -> Dict[str, str]:
    meta_key = f"{prefix}:pdf:{doc_id}:meta"
    if not hasattr(redis_client, "hgetall"):
        return {}
    raw_meta = redis_client.hgetall(meta_key) or {}
    return {str(_decode_redis_value(k)): str(_decode_redis_value(v)) for k, v in raw_meta.items()}


def _doc_partition_chunks(
    redis_client: Any,
    prefix: str,
    doc_id: str,
) -> Tuple[List[Any], List[Any]]:
    meta = _doc_meta(redis_client, prefix, doc_id)
    partitions_key = meta.get("partitions_key") or f"{prefix}:pdf:{doc_id}:partitions"
    chunks_key = meta.get("chunks_key") or f"{prefix}:pdf:{doc_id}:chunks"
    partitions = _safe_json_list(redis_client.get(partitions_key))
    chunks = _safe_json_list(redis_client.get(chunks_key))
    return partitions, chunks


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def _extract_text_preview(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "chunk_text", "content", "chunk", "page_content", "raw_text"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for candidate in value.values():
            nested = _extract_text_preview(candidate)
            if nested:
                return nested
        return ""
    if isinstance(value, list):
        snippets: List[str] = []
        for item in value:
            nested = _extract_text_preview(item)
            if nested:
                snippets.append(nested)
            if len(snippets) >= 4:
                break
        return " ".join(snippets).strip()
    return ""


def _redis_partition_chunk_summary(redis_client: Any, bucket: str, object_name: str) -> Dict[str, Any]:
    source_prefix = _source_redis_prefix()
    source = f"{bucket}/{object_name}"
    doc_ids = _source_doc_ids(redis_client, source_prefix, source)

    partition_count = 0
    chunk_count = 0
    for doc_id in doc_ids:
        partitions, chunks = _doc_partition_chunks(redis_client, source_prefix, doc_id)
        partition_count += len(partitions)
        chunk_count += len(chunks)

    return {
        "source_prefix": source_prefix,
        "source": source,
        "doc_ids": doc_ids,
        "partition_count": partition_count,
        "chunk_count": chunk_count,
    }


def _redis_partition_chunk_items(
    redis_client: Any,
    bucket: str,
    object_name: str,
    kind: str,
    limit: int,
) -> Dict[str, Any]:
    if kind not in {"partitions", "chunks"}:
        raise ValueError("kind must be 'partitions' or 'chunks'")

    source_prefix = _source_redis_prefix()
    source = f"{bucket}/{object_name}"
    doc_ids = _source_doc_ids(redis_client, source_prefix, source)

    total_available = 0
    items: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        partitions, chunks = _doc_partition_chunks(redis_client, source_prefix, doc_id)
        entries = partitions if kind == "partitions" else chunks
        total_available += len(entries)
        for idx, entry in enumerate(entries, start=1):
            text = _truncate_text(_extract_text_preview(entry))
            if not text:
                text = _truncate_text(json.dumps(entry, ensure_ascii=True))
            label = f"Doc {doc_id} - {kind[:-1].title()} {idx}"
            if kind == "chunks" and isinstance(entry, dict):
                chunk_index = entry.get("chunk_index")
                page_start = entry.get("page_start")
                page_end = entry.get("page_end")
                details: List[str] = []
                if chunk_index is not None:
                    details.append(f"chunk {chunk_index}")
                if page_start is not None and page_end is not None:
                    details.append(f"pages {page_start}-{page_end}")
                elif page_start is not None:
                    details.append(f"page {page_start}")
                if details:
                    label = f"{label} ({', '.join(details)})"
            items.append({"doc_id": doc_id, "index": idx, "label": label, "text": text})
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    return {
        "source_prefix": source_prefix,
        "source": source,
        "doc_ids": doc_ids,
        "kind": kind,
        "items": items,
        "count": len(items),
        "total_available": total_available,
        "truncated": len(items) < total_available,
    }


def _default_citation_key(object_name: str) -> str:
    stem = Path(object_name).stem.lower()
    citation_key = re.sub(r"[^a-z0-9]+", "", stem)
    return citation_key[:80] or "untitled"


def _default_metadata(bucket: str, object_name: str) -> Dict[str, Any]:
    return {
        "citationKey": _default_citation_key(object_name),
        "entryType": "article",
        "title": "",
        "year": "",
        "authors": [],
        "journal": "",
        "booktitle": "",
        "volume": "",
        "number": "",
        "pages": "",
        "doi": "",
        "url": "",
        "keywords": "",
        "abstract": "",
        "note": "",
    }


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_authors(value: Any) -> List[Dict[str, str]]:
    authors: List[Dict[str, str]] = []

    if isinstance(value, str):
        if " and " in value:
            parts = [part.strip() for part in value.split(" and ") if part.strip()]
        else:
            parts = [part.strip() for part in value.split(",") if part.strip()]
        for entry in parts:
            authors.append({"firstName": "", "lastName": entry})
        return authors

    if not isinstance(value, list):
        return authors

    for item in value:
        if isinstance(item, dict):
            first_name = _normalize_text(item.get("firstName"))
            last_name = _normalize_text(item.get("lastName"))
            if first_name or last_name:
                authors.append({"firstName": first_name, "lastName": last_name})
            continue
        if isinstance(item, str):
            name = item.strip()
            if name:
                authors.append({"firstName": "", "lastName": name})

    return authors


def _normalize_metadata(
    bucket: str,
    object_name: str,
    metadata: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload = _default_metadata(bucket, object_name)
    if not isinstance(metadata, dict):
        return payload

    for key in (
        "citationKey",
        "title",
        "year",
        "journal",
        "booktitle",
        "volume",
        "number",
        "pages",
        "doi",
        "url",
        "keywords",
        "abstract",
        "note",
    ):
        value = _normalize_text(metadata.get(key))
        if value:
            payload[key] = value
        elif key in {"title", "citationKey"} and payload.get(key):
            # Keep defaults for critical fields.
            continue
        else:
            payload[key] = value

    entry_type = _normalize_text(metadata.get("entryType")).lower()
    if entry_type in ALLOWED_ENTRY_TYPES:
        payload["entryType"] = entry_type

    payload["authors"] = _normalize_authors(metadata.get("authors"))
    return payload


def _load_json_metadata(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_file_metadata(
    redis_client: Any | None,
    prefix: str,
    bucket: str,
    object_name: str,
) -> Dict[str, Any]:
    if not redis_client:
        return _default_metadata(bucket, object_name)

    key = _bibtex_file_key(prefix, bucket, object_name)
    raw = _decode_redis_value(redis_client.get(key))
    payload = _load_json_metadata(raw)

    if not payload and hasattr(redis_client, "hgetall"):
        raw_hash = redis_client.hgetall(key) or {}
        payload = {_decode_redis_value(k): _decode_redis_value(v) for k, v in raw_hash.items()}

    return _normalize_metadata(bucket, object_name, payload)


def _batch_get_file_metadata(
    redis_client: Any | None,
    prefix: str,
    bucket: str,
    object_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    if not redis_client:
        return {object_name: _default_metadata(bucket, object_name) for object_name in object_names}

    keys = [_bibtex_file_key(prefix, bucket, object_name) for object_name in object_names]
    raw_values: List[Any] = []

    if hasattr(redis_client, "pipeline"):
        pipe = redis_client.pipeline()
        for key in keys:
            pipe.get(key)
        raw_values = pipe.execute()
    else:
        raw_values = [redis_client.get(key) for key in keys]

    out: Dict[str, Dict[str, Any]] = {}
    for object_name, raw in zip(object_names, raw_values):
        payload = _load_json_metadata(_decode_redis_value(raw))
        out[object_name] = _normalize_metadata(bucket, object_name, payload)
    return out


def _file_record(bucket: str, object_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    source_ref = build_source_ref(bucket=bucket, key=object_name)
    original_file_url = build_citation_url(source_ref=source_ref)
    return {
        "id": f"{bucket}/{object_name}",
        "bucket": bucket,
        "objectName": object_name,
        "fileName": Path(object_name).name,
        "sourceRef": source_ref,
        "originalFileUrl": original_file_url,
        **metadata,
    }


def _list_bucket_files(
    minio_client: Any,
    redis_client: Any | None,
    bucket: str,
    *,
    limit: int,
    object_prefix: str,
    suffix: str,
) -> List[Dict[str, Any]]:
    prefix = _bibtex_prefix()
    object_names = _list_minio_objects(
        minio_client=minio_client,
        bucket=bucket,
        prefix=object_prefix,
        suffix=suffix,
        limit=limit,
    )
    metadata_by_name = _batch_get_file_metadata(redis_client, prefix, bucket, object_names)
    return [_file_record(bucket, object_name, metadata_by_name[object_name]) for object_name in object_names]


def _save_file_metadata(
    redis_client: Any,
    bucket: str,
    object_name: str,
    payload: Dict[str, Any] | None,
) -> Dict[str, Any]:
    prefix = _bibtex_prefix()
    metadata = _normalize_metadata(bucket, object_name, payload)
    key = _bibtex_file_key(prefix, bucket, object_name)
    redis_client.set(key, json.dumps(metadata, ensure_ascii=True))
    redis_client.sadd(f"{prefix}:files", f"{bucket}/{object_name}")
    return metadata


def _payload_int(payload: Dict[str, Any], key: str, default: int, min_value: int, max_value: int) -> int:
    raw_value = payload.get(key, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = default
    if value < min_value:
        value = min_value
    if value > max_value:
        value = max_value
    return value


def _payload_bool(payload: Dict[str, Any], key: str, default: bool = False) -> bool:
    raw_value = payload.get(key)
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedupe_object_names(object_names: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for entry in object_names:
        name = entry.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _autofill_object_names(
    minio_client: Any,
    bucket: str,
    payload: Dict[str, Any],
) -> List[str]:
    provided = payload.get("objectNames")
    if isinstance(provided, list):
        object_names = [str(entry).strip() for entry in provided if str(entry).strip()]
        suffix = str(payload.get("suffix", os.getenv("BIBTEX_MINIO_SUFFIX", ".pdf"))).strip()
        if suffix:
            object_names = [entry for entry in object_names if entry.lower().endswith(suffix.lower())]
        return _dedupe_object_names(object_names)

    default_prefix = os.getenv("BIBTEX_MINIO_PREFIX", os.getenv("MINIO_PREFIX", "")).strip()
    default_suffix = os.getenv("BIBTEX_MINIO_SUFFIX", ".pdf").strip()
    object_prefix = str(payload.get("objectPrefix", default_prefix))
    suffix = str(payload.get("suffix", default_suffix))
    limit = _payload_int(payload, "limit", 10000, 1, 50000)
    return _list_minio_objects(
        minio_client=minio_client,
        bucket=bucket,
        prefix=object_prefix,
        suffix=suffix,
        limit=limit,
    )


def _crossref_client_from_env():
    if bibtex_autofill is None:
        raise RuntimeError("bibtex_autofill module is unavailable")
    try:
        timeout_seconds = int(os.getenv("CROSSREF_TIMEOUT_SECONDS", "20"))
    except ValueError:
        timeout_seconds = 20
    try:
        rows = int(os.getenv("CROSSREF_ROWS", "5"))
    except ValueError:
        rows = 5
    try:
        throttle_seconds = float(os.getenv("CROSSREF_THROTTLE_SECONDS", "0.15"))
    except ValueError:
        throttle_seconds = 0.15
    return bibtex_autofill.CrossrefClient(
        api_url=os.getenv("CROSSREF_API_URL", "https://api.crossref.org"),
        timeout_seconds=timeout_seconds,
        rows=rows,
        mailto=os.getenv("CROSSREF_MAILTO", ""),
        user_agent=os.getenv("CROSSREF_USER_AGENT", ""),
        throttle_seconds=throttle_seconds,
    )


def _autofill_missing_metadata_batch(
    *,
    minio_client: Any,
    redis_client: Any,
    bucket: str,
    object_names: List[str],
    offset: int,
    batch_size: int,
    dry_run: bool,
) -> Dict[str, Any]:
    if bibtex_autofill is None:
        raise RuntimeError("bibtex_autofill module is unavailable")

    total = len(object_names)
    safe_offset = max(0, min(offset, total))
    safe_batch_size = max(1, batch_size)
    end = min(total, safe_offset + safe_batch_size)
    batch = object_names[safe_offset:end]

    counts = {
        "updated": 0,
        "dry_run_update": 0,
        "skipped_existing": 0,
        "no_match": 0,
        "low_confidence": 0,
        "doi_conflict": 0,
        "no_signals": 0,
        "error": 0,
    }
    results: List[Dict[str, Any]] = []
    crossref_client = _crossref_client_from_env()

    for object_name in batch:
        try:
            result = bibtex_autofill.enrich_file_metadata(
                minio_client=minio_client,
                redis_client=redis_client,
                bucket=bucket,
                object_name=object_name,
                bibtex_prefix=_bibtex_prefix(),
                source_prefix=_source_redis_prefix(),
                overwrite=False,
                dry_run=dry_run,
                skip_complete=True,
                crossref_client=crossref_client,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            result = {"bucket": bucket, "object_name": object_name, "status": "error", "error": str(exc)}

        status = str(result.get("status", "error"))
        counts[status] = counts.get(status, 0) + 1

        out_entry = {
            "objectName": object_name,
            "status": status,
            "error": result.get("error"),
            "match": result.get("match") if isinstance(result.get("match"), dict) else {},
            "candidates": result.get("candidates") if isinstance(result.get("candidates"), list) else [],
        }
        if status in {"updated", "dry_run_update", "skipped_existing"}:
            metadata = _get_file_metadata(redis_client, _bibtex_prefix(), bucket, object_name)
            out_entry["file"] = _file_record(bucket, object_name, metadata)
        results.append(out_entry)

    processed_total = end
    done = processed_total >= total
    return {
        "total": total,
        "offset": safe_offset,
        "batch_size": safe_batch_size,
        "processed_in_batch": len(batch),
        "processed_total": processed_total,
        "next_offset": None if done else processed_total,
        "done": done,
        "counts": counts,
        "results": results,
    }


load_dotenv()

app = FastAPI(title="MCP BibTeX UI") if FastAPI is not None else None


if app is not None:

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_load_ui_html())

    @app.get("/api/status")
    def api_status() -> Dict[str, Any]:
        minio_client, minio_error = _get_minio_client()
        redis_client, redis_error = _get_redis_client()
        bucket_names: List[str] = []
        bucket_error = None
        if minio_client:
            bucket_names, bucket_error = _resolve_minio_buckets(minio_client)
        return {
            "minio": {"connected": bool(minio_client), "error": minio_error, "bucket_error": bucket_error},
            "redis": {"connected": bool(redis_client), "error": redis_error},
            "bibtex_prefix": _bibtex_prefix(),
            "buckets": bucket_names,
        }

    @app.get("/api/buckets")
    def api_buckets() -> Dict[str, Any]:
        minio_client, minio_error = _get_minio_client()
        if minio_error or not minio_client:
            raise HTTPException(status_code=503, detail=minio_error or "MinIO is not configured")
        bucket_names, bucket_error = _resolve_minio_buckets(minio_client)
        if bucket_error:
            raise HTTPException(status_code=400, detail=bucket_error)
        return {"buckets": bucket_names}

    @app.get("/api/buckets/{bucket}/files")
    def api_bucket_files(
        bucket: str,
        limit: int = Query(1000, ge=1, le=20000),
        object_prefix: str | None = None,
        suffix: str | None = None,
    ) -> Dict[str, Any]:
        minio_client, minio_error = _get_minio_client()
        if minio_error or not minio_client:
            raise HTTPException(status_code=503, detail=minio_error or "MinIO is not configured")
        redis_client, _redis_error = _get_redis_client()

        default_prefix = os.getenv("BIBTEX_MINIO_PREFIX", os.getenv("MINIO_PREFIX", "")).strip()
        default_suffix = os.getenv("BIBTEX_MINIO_SUFFIX", ".pdf").strip()

        try:
            files = _list_bucket_files(
                minio_client=minio_client,
                redis_client=redis_client,
                bucket=bucket,
                limit=limit,
                object_prefix=object_prefix if object_prefix is not None else default_prefix,
                suffix=suffix if suffix is not None else default_suffix,
            )
        except S3Error as exc:
            detail = str(exc)
            if "NoSuchBucket" in detail:
                raise HTTPException(status_code=404, detail=f"MinIO bucket not found: {bucket}") from exc
            raise HTTPException(status_code=500, detail=f"Failed to list objects for bucket {bucket}: {exc}") from exc
        return {"bucket": bucket, "count": len(files), "files": files}

    @app.get("/api/buckets/{bucket}/files/{object_name:path}/bibtex")
    def api_file_bibtex(bucket: str, object_name: str) -> Dict[str, Any]:
        redis_client, redis_error = _get_redis_client()
        if redis_error or not redis_client:
            raise HTTPException(status_code=503, detail=redis_error or "Redis is not configured")
        metadata = _get_file_metadata(redis_client, _bibtex_prefix(), bucket, object_name)
        return {"bucket": bucket, "objectName": object_name, "file": _file_record(bucket, object_name, metadata)}

    @app.put("/api/buckets/{bucket}/files/{object_name:path}/bibtex")
    def api_save_file_bibtex(bucket: str, object_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        redis_client, redis_error = _get_redis_client()
        if redis_error or not redis_client:
            raise HTTPException(status_code=503, detail=redis_error or "Redis is not configured")
        metadata = _save_file_metadata(redis_client, bucket, object_name, payload)
        return {
            "bucket": bucket,
            "objectName": object_name,
            "redis_key": _bibtex_file_key(_bibtex_prefix(), bucket, object_name),
            "file": _file_record(bucket, object_name, metadata),
        }

    @app.get("/api/buckets/{bucket}/files/{object_name:path}/redis-summary")
    def api_file_redis_summary(bucket: str, object_name: str) -> Dict[str, Any]:
        redis_client, redis_error = _get_redis_client()
        if redis_error or not redis_client:
            raise HTTPException(status_code=503, detail=redis_error or "Redis is not configured")
        summary = _redis_partition_chunk_summary(redis_client, bucket, object_name)
        return {"bucket": bucket, "objectName": object_name, **summary}

    @app.get("/api/buckets/{bucket}/files/{object_name:path}/redis-data")
    def api_file_redis_data(
        bucket: str,
        object_name: str,
        kind: str = Query("chunks"),
        limit: int = Query(200, ge=1, le=5000),
    ) -> Dict[str, Any]:
        redis_client, redis_error = _get_redis_client()
        if redis_error or not redis_client:
            raise HTTPException(status_code=503, detail=redis_error or "Redis is not configured")
        try:
            payload = _redis_partition_chunk_items(redis_client, bucket, object_name, kind=kind, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"bucket": bucket, "objectName": object_name, **payload}

    @app.post("/api/buckets/{bucket}/autofill-missing")
    def api_bucket_autofill_missing(bucket: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        minio_client, minio_error = _get_minio_client()
        if minio_error or not minio_client:
            raise HTTPException(status_code=503, detail=minio_error or "MinIO is not configured")
        redis_client, redis_error = _get_redis_client()
        if redis_error or not redis_client:
            raise HTTPException(status_code=503, detail=redis_error or "Redis is not configured")

        try:
            object_names = _autofill_object_names(minio_client, bucket, payload or {})
        except S3Error as exc:
            raise HTTPException(status_code=500, detail=f"Failed to list objects for bucket {bucket}: {exc}") from exc

        offset = _payload_int(payload or {}, "offset", 0, 0, max(len(object_names), 0))
        batch_size = _payload_int(payload or {}, "batchSize", 25, 1, 500)
        dry_run = _payload_bool(payload or {}, "dryRun", False)

        try:
            report = _autofill_missing_metadata_batch(
                minio_client=minio_client,
                redis_client=redis_client,
                bucket=bucket,
                object_names=object_names,
                offset=offset,
                batch_size=batch_size,
                dry_run=dry_run,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {"bucket": bucket, **report}


def main() -> None:
    import uvicorn  # pylint: disable=import-outside-toplevel

    if app is None:  # pragma: no cover
        raise RuntimeError("fastapi is required to run the BibTeX UI")

    host = os.getenv("BIBTEX_UI_HOST", "0.0.0.0")
    port = int(os.getenv("BIBTEX_UI_PORT", "8003"))
    uvicorn.run("mcp_research.bibtex_ui_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
