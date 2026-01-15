import json
import os
from html import escape
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from qdrant_client import QdrantClient, models

from mcp_research.link_resolver import resolve_link

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None

app = FastAPI(title="Citation Link Resolver", version="0.1.0")


def _decode_redis_value(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _get_redis_client():
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url or redis is None:
        return None
    return redis.from_url(redis_url)


def _get_qdrant_client() -> QdrantClient | None:
    qdrant_url = os.getenv("QDRANT_URL", "")
    if not qdrant_url:
        return None
    return QdrantClient(url=qdrant_url)


def _derive_bucket_and_key(source: str) -> Tuple[str, str]:
    if "/" in source:
        bucket, key = source.split("/", 1)
        return bucket, key
    bucket = os.getenv("SOURCE_BUCKET", "local")
    return bucket, source


def _load_partition_count(redis_client, partitions_key: str) -> int:
    raw = _decode_redis_value(redis_client.get(partitions_key))
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _load_chunk_count(redis_client, chunks_key: str) -> int:
    raw = _decode_redis_value(redis_client.get(chunks_key))
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _qdrant_uploaded(client: QdrantClient | None, collection: str, document_id: str) -> str:
    if not client:
        return "unknown"
    try:
        if not client.collection_exists(collection):
            return "no"
        response = client.count(
            collection_name=collection,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            ),
            exact=True,
        )
    except Exception:
        return "unknown"
    return "yes" if response.count > 0 else "no"


def _load_inventory() -> Tuple[Dict[str, List[dict]], str | None]:
    redis_client = _get_redis_client()
    if not redis_client:
        return {}, "REDIS_URL is required for the inventory dashboard"
    redis_prefix = os.getenv("REDIS_PREFIX", "unstructured")
    qdrant_client = _get_qdrant_client()

    raw_ids = redis_client.smembers(f"{redis_prefix}:pdf:hashes")
    doc_ids = [_decode_redis_value(entry) for entry in raw_ids or []]

    buckets: Dict[str, List[dict]] = {}
    for doc_id in sorted(filter(None, doc_ids)):
        meta_key = f"{redis_prefix}:pdf:{doc_id}:meta"
        meta_raw = redis_client.hgetall(meta_key)
        meta = { _decode_redis_value(k): _decode_redis_value(v) for k, v in meta_raw.items() }
        source = meta.get("source") or doc_id
        bucket, key = _derive_bucket_and_key(source)
        partitions_key = meta.get("partitions_key") or f"{redis_prefix}:pdf:{doc_id}:partitions"
        chunks_key = meta.get("chunks_key") or f"{redis_prefix}:pdf:{doc_id}:chunks"
        collections_key = meta.get("collections_key") or f"{redis_prefix}:pdf:{doc_id}:collections"

        partitions_count = _load_partition_count(redis_client, partitions_key)
        chunks_count = meta.get("chunks")
        chunks_count_value = int(chunks_count) if str(chunks_count).isdigit() else None
        if chunks_count_value is None:
            chunks_count_value = _load_chunk_count(redis_client, chunks_key)

        collections = [
            _decode_redis_value(entry)
            for entry in (redis_client.smembers(collections_key) or [])
        ]
        expected_collection = bucket if "/" in source else os.getenv("QDRANT_COLLECTION", "pdf_chunks")
        qdrant_status = _qdrant_uploaded(qdrant_client, expected_collection, doc_id)

        buckets.setdefault(bucket, []).append(
            {
                "doc_id": doc_id,
                "key": key,
                "source": source,
                "partitions": partitions_count,
                "chunks": chunks_count_value,
                "collections": [c for c in collections if c],
                "expected_collection": expected_collection,
                "qdrant_status": qdrant_status,
            }
        )
    return buckets, None


def _render_dashboard(buckets: Dict[str, List[dict]], error: str | None) -> str:
    body = []
    if error:
        body.append(f"<div class='notice'>{escape(error)}</div>")
    if not buckets:
        body.append("<div class='empty'>No PDFs found in Redis.</div>")
    for bucket, entries in buckets.items():
        body.append(f"<section class='bucket'><h2>{escape(bucket)}</h2>")
        body.append("<table><thead><tr>"
                    "<th>PDF</th><th>Partitions</th><th>Chunks</th>"
                    "<th>Qdrant</th><th>Collection</th></tr></thead><tbody>")
        for entry in sorted(entries, key=lambda item: item["key"]):
            status = entry["qdrant_status"]
            status_label = "unknown" if status == "unknown" else ("yes" if status == "yes" else "no")
            body.append(
                "<tr>"
                f"<td><div class='file'>{escape(entry['key'])}</div>"
                f"<div class='meta'>{escape(entry['doc_id'])}</div></td>"
                f"<td>{entry['partitions']}</td>"
                f"<td>{entry['chunks']}</td>"
                f"<td><span class='status {status_label}'>{status_label}</span></td>"
                f"<td>{escape(entry['expected_collection'])}</td>"
                "</tr>"
            )
        body.append("</tbody></table></section>")

    content = "\n".join(body)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PDF Inventory</title>
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
    .notice {{
      max-width: 1100px;
      margin: 0 auto 20px;
      padding: 12px 16px;
      border-radius: 12px;
      background: #fff0f0;
      color: #8a2f2f;
      border: 1px solid #f1b3b3;
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
    <h1>PDF Inventory</h1>
    <p>Grouped by bucket with partitions, chunks, and Qdrant upload status.</p>
  </header>
  {content}
</body>
</html>"""


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
def inventory_ui():
    buckets, error = _load_inventory()
    html = _render_dashboard(buckets, error)
    return HTMLResponse(content=html)


def main() -> None:
    host = os.getenv("RESOLVER_HOST", "0.0.0.0")
    port = int(os.getenv("RESOLVER_PORT", "8080"))
    import uvicorn

    uvicorn.run("mcp_research.resolver_app:app", host=host, port=port)


if __name__ == "__main__":
    main()
