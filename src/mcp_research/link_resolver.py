import os
from datetime import timedelta
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from minio import Minio
from mcp_research.citation_utils import build_citation_url, build_source_ref


def _normalize_source_ref(source_ref: str) -> str:
    """Normalize incoming source references from URLs or encoded strings."""
    if not source_ref:
        return source_ref
    raw = source_ref.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        query = parse_qs(parsed.query or "")
        ref_values = query.get("ref", [])
        if ref_values:
            return ref_values[0]
    if raw.startswith("ref="):
        return raw.split("=", 1)[1]
    if raw.startswith("doc:%2F%2F"):
        return unquote(raw)
    return raw


def parse_source_ref(source_ref: str) -> dict:
    """Parse a doc:// source reference into its component parts."""
    normalized = _normalize_source_ref(source_ref)
    parsed = urlparse(normalized)
    if parsed.scheme != "doc":
        raise ValueError(f"Unsupported source_ref scheme: {parsed.scheme}")
    bucket = parsed.netloc
    key = unquote(parsed.path.lstrip("/"))
    query = parse_qs(parsed.query or "")
    version_id_values = query.get("version_id", [])
    version_id = version_id_values[0] if version_id_values else None

    page_start = None
    page_end = None
    fragment = parsed.fragment or ""
    if fragment.startswith("page="):
        page_value = fragment.split("=", 1)[1]
        if "-" in page_value:
            start_raw, end_raw = page_value.split("-", 1)
            try:
                page_start = int(start_raw)
                page_end = int(end_raw)
            except ValueError:
                page_start = None
                page_end = None
        else:
            try:
                page_start = int(page_value)
                page_end = page_start
            except ValueError:
                page_start = None
                page_end = None

    return {
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "page_start": page_start,
        "page_end": page_end,
    }


def _get_minio_client() -> Minio:
    """Create a MinIO client for generating presigned URLs."""
    endpoint = os.getenv("MINIO_PRESIGN_ENDPOINT") or os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    secure_value = os.getenv("MINIO_PRESIGN_SECURE") or os.getenv("MINIO_SECURE", "false")
    secure = secure_value.strip().lower() in {"1", "true", "yes", "y", "on"}
    region = os.getenv("MINIO_PRESIGN_REGION") or os.getenv("MINIO_REGION") or (
        "us-east-1" if os.getenv("MINIO_PRESIGN_ENDPOINT") else None
    )
    if not access_key or not secret_key:
        raise RuntimeError("MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required for presigned URLs")
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
        region=region,
    )


def _build_pdf_fragment(page_start: Optional[int], page_end: Optional[int], highlight: Optional[str]) -> str:
    """Build a PDF URL fragment with page targeting and optional text search."""
    fragment_parts = []
    if page_start is None and page_end is not None:
        page_start = page_end
    if page_start is not None:
        page = page_start if page_end is None else min(page_start, page_end)
        fragment_parts.append(f"page={page}")

    text = (highlight or "").strip()
    if text:
        fragment_parts.append(f"search={quote(text, safe='')}")

    return "&".join(fragment_parts)


def _append_url_fragment(url: str, fragment: str) -> str:
    """Append or merge a URL fragment while preserving existing URL parts."""
    if not fragment:
        return url

    parsed = urlparse(url)
    if not parsed.fragment:
        return parsed._replace(fragment=fragment).geturl()

    existing_parts = [part for part in parsed.fragment.split("&") if part]
    extra_parts = [part for part in fragment.split("&") if part and part not in existing_parts]
    merged = "&".join(existing_parts + extra_parts)
    return parsed._replace(fragment=merged).geturl()


def _append_url_query(url: str, params: dict[str, Optional[str]]) -> str:
    """Append query params when missing, preserving existing query values."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query or "", keep_blank_values=True)
    for key, value in params.items():
        if value is None or key in query:
            continue
        query[key] = [value]
    encoded = urlencode(query, doseq=True)
    return parsed._replace(query=encoded).geturl()


def _build_pdf_proxy_url(source_ref: str) -> str:
    """Build a resolver-local PDF proxy URL for a source reference."""
    base = (
        os.getenv("CITATION_BASE_URL")
        or os.getenv("DOCS_BASE_URL")
        or "http://localhost:8080"
    ).rstrip("/")
    proxy_path = os.getenv("CITATION_PDF_PROXY_PATH", "/r/pdf-proxy")
    if not proxy_path.startswith("/"):
        proxy_path = "/" + proxy_path
    return f"{base}{proxy_path}?ref={quote(source_ref, safe='')}"


def resolve_link(
    source_ref: Optional[str] = None,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    version_id: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    page: Optional[int] = None,
    highlight: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict:
    """Resolve a source reference into a portal, CDN, or presigned URL."""
    if source_ref:
        parts = parse_source_ref(source_ref)
        bucket = bucket or parts.get("bucket")
        key = key or parts.get("key")
        version_id = version_id or parts.get("version_id")
        if page_start is None and page_end is None:
            page_start = parts.get("page_start")
            page_end = parts.get("page_end")
    if page is not None:
        page_start = page
        page_end = page
    if not source_ref:
        if not bucket or not key:
            raise ValueError("bucket/key or source_ref are required")
        source_ref = build_source_ref(
            bucket=bucket,
            key=key,
            page_start=page_start,
            page_end=page_end,
            version_id=version_id,
        )

    resolved_mode = (mode or os.getenv("LINK_RESOLVER_MODE", "portal")).strip().lower()
    if resolved_mode == "presign":
        minio_client = _get_minio_client()
        expiry = int(os.getenv("MINIO_PRESIGN_EXPIRY_SECONDS", "3600"))
        url = minio_client.presigned_get_object(
            bucket_name=bucket,
            object_name=key,
            expires=timedelta(seconds=expiry),
            version_id=version_id,
        )
        url = _append_url_fragment(url, _build_pdf_fragment(page_start, page_end, highlight))
        return {"source_ref": source_ref, "url": url, "mode": resolved_mode}
    if resolved_mode == "cdn":
        base = os.getenv("CDN_BASE_URL", "").rstrip("/")
        if not base:
            raise RuntimeError("CDN_BASE_URL is required for CDN link resolution")
        safe_key = quote(key.lstrip("/"), safe="/")
        url = _append_url_fragment(f"{base}/{safe_key}", _build_pdf_fragment(page_start, page_end, highlight))
        return {"source_ref": source_ref, "url": url, "mode": resolved_mode}
    if resolved_mode in {"proxy", "pdf-proxy"}:
        url = _build_pdf_proxy_url(source_ref)
        url = _append_url_fragment(url, _build_pdf_fragment(page_start, page_end, highlight))
        return {"source_ref": source_ref, "url": url, "mode": "proxy"}

    url = build_citation_url(source_ref)
    if not url:
        raise RuntimeError("CITATION_BASE_URL (or DOCS_BASE_URL) is required for portal links")
    portal_page = str(page_start) if page_start is not None else (str(page_end) if page_end is not None else None)
    portal_page_start = str(page_start) if page_start is not None else None
    portal_page_end = str(page_end) if page_end is not None else None
    portal_highlight = (highlight or "").strip() or None
    url = _append_url_query(
        url,
        {
            "page": portal_page,
            "page_start": portal_page_start,
            "page_end": portal_page_end,
            "highlight": portal_highlight,
        },
    )
    return {"source_ref": source_ref, "url": url, "mode": "portal"}
