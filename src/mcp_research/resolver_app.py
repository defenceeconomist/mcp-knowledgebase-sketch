import json
import os
import urllib.error
import urllib.request
from html import escape
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlparse, urlunparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse, Response
from qdrant_client import QdrantClient, models

from mcp_research.link_resolver import parse_source_ref, resolve_link
from mcp_research.runtime_utils import (
    decode_redis_value as _decode_redis_value,
    load_env_bool as _load_env_bool,
    load_env_list as _load_env_list,
)
from mcp_research.schema_v2 import read_v2_source_doc_hash, redis_v2_source_meta_key

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
    if hasattr(redis_client, "zcard"):
        return int(redis_client.zcard(partitions_key))
    return 0


def _load_chunk_count(redis_client, chunks_key: str) -> int:
    """Return the number of chunks stored in Redis for a document."""
    if hasattr(redis_client, "zcard"):
        return int(redis_client.zcard(chunks_key))
    return 0


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
            must.append(models.FieldCondition(key="doc_hash", match=models.MatchValue(value=document_id)))
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
    bucket, key = _split_source(source)
    if not bucket or not key:
        return []
    doc_hash = read_v2_source_doc_hash(
        redis_client=redis_client,
        prefix=redis_prefix,
        bucket=bucket,
        key=key,
        version_id=None,
    )
    return [doc_hash] if doc_hash else []


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
        meta_key = f"{redis_prefix}:v2:doc:{doc_id}:meta"
        if redis_client.exists(meta_key):
            return "yes"
    if source:
        doc_ids = _source_doc_ids(redis_client, redis_prefix, source)
        if doc_id:
            return "yes" if doc_id in doc_ids else "no"
        return "yes" if doc_ids else "no"
    return "no"


def _build_redis_source_index(redis_client, redis_prefix: str) -> Dict[str, str]:
    """Build a document_id -> source index from Redis metadata."""
    index: Dict[str, str] = {}
    if not redis_client:
        return index
    raw_doc_ids = redis_client.smembers(f"{redis_prefix}:v2:doc_hashes") or []
    doc_ids = filter(None, (_decode_redis_value(val) for val in raw_doc_ids))
    for doc_id in doc_ids:
        meta_key = f"{redis_prefix}:v2:doc:{doc_id}:meta"
        meta_raw = redis_client.hgetall(meta_key)
        meta = {_decode_redis_value(k): _decode_redis_value(v) for k, v in meta_raw.items()}
        primary_sid = meta.get("primary_source_id")
        source = None
        if primary_sid:
            source_meta_raw = redis_client.hgetall(redis_v2_source_meta_key(redis_prefix, str(primary_sid)))
            source_meta = {_decode_redis_value(k): _decode_redis_value(v) for k, v in source_meta_raw.items()}
            source = source_meta.get("source_path")
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
                doc_id = payload.get("doc_hash") or payload.get("document_id")
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


def _with_pdf_fragment_params(url: str, params: Dict[str, str]) -> str:
    """Merge PDF fragment params into a URL without dropping existing entries."""
    parsed = urlparse(url)
    existing = parse_qsl(parsed.fragment or "", keep_blank_values=True)
    merged: Dict[str, str] = {}
    for key, value in existing:
        if key and key not in merged:
            merged[key] = value
    for key, value in (params or {}).items():
        if not key or key in merged or value is None:
            continue
        merged[key] = str(value)
    fragment = "&".join(f"{quote(key, safe='')}={quote(value, safe='')}" for key, value in merged.items())
    return urlunparse(parsed._replace(fragment=fragment))


def _strip_url_fragment(url: str) -> str:
    """Return a URL without a fragment section."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment=""))


def _render_pdf_embed_page(url: str, page: int, highlight: str, source_ref: str | None = None) -> str:
    """Render a PDF.js-based citation viewer with deterministic highlight behavior."""
    clean_url = _strip_url_fragment(url)
    if source_ref:
        proxy_url = f"/r/pdf-proxy?ref={quote(source_ref, safe='')}"
    else:
        proxy_url = f"/r/pdf-proxy?url={quote(clean_url, safe='')}"
    safe_direct_url = escape(url, quote=True)
    payload = json.dumps(
        {
            "pdfUrl": proxy_url,
            "page": int(page or 1),
            "highlight": str(highlight or ""),
        },
        ensure_ascii=True,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Citation Viewer</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #f4f4f4;
      color: #1f2328;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }}
    .shell {{
      position: fixed;
      inset: 0;
      display: flex;
      flex-direction: column;
    }}
    .bar {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 12px;
      min-height: 40px;
      font-size: 12px;
      border-bottom: 1px solid #d6d9de;
      background: #ffffff;
    }}
    .bar a {{
      color: #0b5cab;
      text-decoration: none;
      font-weight: 600;
    }}
    .status {{
      color: #52606d;
      margin-left: auto;
      font-size: 11px;
    }}
    .viewer {{
      flex: 1;
      min-height: 0;
      overflow: auto;
      background: #eceff3;
      padding: 16px;
    }}
    .page-wrap {{
      position: relative;
      margin: 0 auto;
      width: fit-content;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
      background: #fff;
    }}
    #pdfCanvas {{
      display: block;
      max-width: 100%;
      height: auto;
    }}
    .textLayer {{
      position: absolute;
      inset: 0;
      overflow: hidden;
      line-height: 1;
      -webkit-text-size-adjust: none;
      text-size-adjust: none;
    }}
    .textLayer span {{
      color: transparent;
      position: absolute;
      white-space: pre;
      transform-origin: 0 0;
    }}
    .textLayer .hl {{
      background: rgba(255, 228, 92, 0.7);
      border-radius: 2px;
      animation: fade-hl 4.2s ease forwards;
    }}
    @keyframes fade-hl {{
      0% {{ background: rgba(255, 228, 92, 0.9); }}
      100% {{ background: rgba(255, 228, 92, 0.18); }}
    }}
    .fallback-frame {{
      width: 100%;
      height: 100%;
      border: none;
      display: none;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="bar">
      <span>Citation Viewer</span>
      <a href="{safe_direct_url}" target="_blank" rel="noopener noreferrer">Open direct PDF</a>
      <span class="status" id="viewerStatus">Loading PDF...</span>
    </div>
    <div class="viewer" id="viewerRoot">
      <div class="page-wrap" id="pageWrap">
        <canvas id="pdfCanvas"></canvas>
        <div class="textLayer" id="textLayer"></div>
      </div>
      <iframe class="fallback-frame" id="fallbackFrame" src="{safe_direct_url}" title="Citation PDF fallback"></iframe>
    </div>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <script>
    (function () {{
      const cfg = {payload};
      const statusEl = document.getElementById("viewerStatus");
      const canvas = document.getElementById("pdfCanvas");
      const textLayer = document.getElementById("textLayer");
      const pageWrap = document.getElementById("pageWrap");
      const fallback = document.getElementById("fallbackFrame");
      const viewerRoot = document.getElementById("viewerRoot");

      function setStatus(msg) {{
        if (statusEl) statusEl.textContent = msg;
      }}

      function normalizeText(value) {{
        return String(value || "")
          .toLowerCase()
          .replace(/[^0-9a-z]+/g, " ")
          .replace(/\\s+/g, " ")
          .trim();
      }}

      function markHighlights(term) {{
        const needle = normalizeText(term);
        if (!needle) return 0;
        const spans = textLayer.querySelectorAll("span");
        let count = 0;
        for (const span of spans) {{
          const hay = normalizeText(span.textContent || "");
          if (!hay) continue;
          if (hay.includes(needle) || needle.includes(hay)) {{
            span.classList.add("hl");
            count += 1;
          }}
        }}
        return count;
      }}

      function showFallback(reason) {{
        console.warn("PDF.js fallback:", reason);
        if (pageWrap) pageWrap.style.display = "none";
        if (fallback) fallback.style.display = "block";
        setStatus("Loaded with browser PDF fallback.");
      }}

      async function render() {{
        if (!window.pdfjsLib) {{
          showFallback("pdfjs unavailable");
          return;
        }}
        const workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
        window.pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;
        try {{
          const loadingTask = window.pdfjsLib.getDocument({{ url: cfg.pdfUrl }});
          const pdf = await loadingTask.promise;
          const targetPage = Math.max(1, Math.min(pdf.numPages, Number(cfg.page || 1)));
          const page = await pdf.getPage(targetPage);

          const unscaled = page.getViewport({{ scale: 1 }});
          const rootWidth = Math.max(400, Math.floor((viewerRoot && viewerRoot.clientWidth) ? viewerRoot.clientWidth - 32 : 1000));
          const fitScale = Math.max(1.15, rootWidth / unscaled.width);
          const viewport = page.getViewport({{ scale: fitScale }});

          const dpr = window.devicePixelRatio || 1;
          canvas.width = Math.floor(viewport.width * dpr);
          canvas.height = Math.floor(viewport.height * dpr);
          canvas.style.width = `${{Math.floor(viewport.width)}}px`;
          canvas.style.height = `${{Math.floor(viewport.height)}}px`;
          textLayer.style.width = `${{Math.floor(viewport.width)}}px`;
          textLayer.style.height = `${{Math.floor(viewport.height)}}px`;

          const ctx = canvas.getContext("2d", {{ alpha: false }});
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          await page.render({{ canvasContext: ctx, viewport }}).promise;

          const textContent = await page.getTextContent();
          textLayer.innerHTML = "";
          const textDivs = [];
          const task = window.pdfjsLib.renderTextLayer({{
            textContentSource: textContent,
            container: textLayer,
            viewport,
            textDivs,
          }});
          if (task && task.promise) {{
            await task.promise;
          }} else if (task && typeof task.then === "function") {{
            await task;
          }}

          const marked = markHighlights(cfg.highlight);
          if (marked > 0) {{
            setStatus(`Page ${{targetPage}} loaded. Highlighted ${{marked}} match(es).`);
          }} else if (cfg.highlight) {{
            setStatus(`Page ${{targetPage}} loaded. No text-layer match for "${{cfg.highlight}}".`);
          }} else {{
            setStatus(`Page ${{targetPage}} loaded.`);
          }}
        }} catch (err) {{
          showFallback(err && err.message ? err.message : String(err));
        }}
      }}

      render();
    }})();
  </script>
</body>
</html>"""


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
    raw_doc_ids = redis_client.smembers(f"{redis_prefix}:v2:doc_hashes") or []
    doc_ids = sorted(filter(None, (_decode_redis_value(val) for val in raw_doc_ids)))
    for doc_id in doc_ids:
        meta_key = f"{redis_prefix}:v2:doc:{doc_id}:meta"
        meta_raw = redis_client.hgetall(meta_key)
        meta = {_decode_redis_value(k): _decode_redis_value(v) for k, v in meta_raw.items()}
        source = ""
        primary_sid = meta.get("primary_source_id")
        if primary_sid:
            source_meta_raw = redis_client.hgetall(redis_v2_source_meta_key(redis_prefix, str(primary_sid)))
            source_meta = {_decode_redis_value(k): _decode_redis_value(v) for k, v in source_meta_raw.items()}
            source = source_meta.get("source_path") or ""
        collections_key = meta.get("collections_key") or f"{redis_prefix}:v2:doc:{doc_id}:collections"
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
                    meta_key = f"{redis_prefix}:v2:doc:{doc_id}:meta"
                    meta_raw = redis_client.hgetall(meta_key)
                    meta = { _decode_redis_value(k): _decode_redis_value(v) for k, v in meta_raw.items() }
                    partitions_key = f"{redis_prefix}:v2:doc:{doc_id}:partition_hashes"
                    chunks_key = f"{redis_prefix}:v2:doc:{doc_id}:chunk_hashes"
                    partitions_count = _load_partition_count(redis_client, partitions_key)
                    chunks_count = meta.get("chunks_count")
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
    highlight: Optional[str] = Query(default=None),
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
            highlight=highlight,
            mode=mode,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    focus_requested = any(
        value is not None and str(value).strip() != ""
        for value in (page, page_start, page_end, highlight)
    )
    if result.get("mode") == "presign" and focus_requested:
        target_page = page if page is not None else page_start if page_start is not None else page_end if page_end is not None else 1
        return HTMLResponse(
            content=_render_pdf_embed_page(
                result["url"],
                page=int(target_page or 1),
                highlight=str(highlight or ""),
                source_ref=result.get("source_ref"),
            ),
            status_code=200,
        )
    return RedirectResponse(url=result["url"], status_code=302)


@app.get("/r/pdf-proxy")
def resolve_pdf_proxy(
    ref: Optional[str] = Query(default=None, description="doc:// source reference"),
    url: Optional[str] = Query(default=None, description="Absolute URL to proxy"),
):
    """Proxy PDF bytes same-origin for deterministic in-browser rendering."""
    if ref:
        try:
            parsed_ref = parse_source_ref(ref)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid ref: {exc}") from exc
        bucket = parsed_ref.get("bucket")
        key = parsed_ref.get("key")
        version_id = parsed_ref.get("version_id")
        if not bucket or not key:
            raise HTTPException(status_code=400, detail="ref must include bucket and key")
        minio_client, minio_error = _get_minio_client()
        if minio_error or not minio_client:
            raise HTTPException(status_code=503, detail=minio_error or "MinIO client unavailable")
        try:
            response = minio_client.get_object(bucket, key, version_id=version_id)
            try:
                payload = response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch object from MinIO: {exc}") from exc
        headers = {"Cache-Control": "no-store"}
        return Response(content=payload, media_type="application/pdf", headers=headers)

    if not url:
        raise HTTPException(status_code=400, detail="Either ref or url is required")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http/https URLs are supported")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "mcp-research-resolver/1.0"})
        with urllib.request.urlopen(request, timeout=90) as upstream:
            payload = upstream.read()
            content_type = upstream.headers.get("Content-Type") or "application/pdf"
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream PDF fetch failed: HTTP {exc.code}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream PDF fetch failed: {exc}") from exc
    headers = {"Cache-Control": "no-store"}
    return Response(content=payload, media_type=content_type, headers=headers)


@app.get("/r/doc.json")
def resolve_doc_json(
    ref: Optional[str] = Query(default=None, description="doc:// reference"),
    bucket: Optional[str] = Query(default=None),
    key: Optional[str] = Query(default=None),
    version_id: Optional[str] = Query(default=None),
    page: Optional[int] = Query(default=None),
    page_start: Optional[int] = Query(default=None),
    page_end: Optional[int] = Query(default=None),
    highlight: Optional[str] = Query(default=None),
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
            highlight=highlight,
            mode=mode,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=result)


def main() -> None:
    """CLI entry point for the resolver API server."""
    host = os.getenv("RESOLVER_HOST", "0.0.0.0")
    port = int(os.getenv("RESOLVER_PORT", "8080"))
    import uvicorn

    uvicorn.run("mcp_research.resolver_app:app", host=host, port=port)


if __name__ == "__main__":
    main()
