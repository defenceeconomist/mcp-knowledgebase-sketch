import os
from datetime import timedelta
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from minio import Minio


def build_source_ref(
    bucket: str,
    key: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    version_id: Optional[str] = None,
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


def parse_source_ref(source_ref: str) -> dict:
    parsed = urlparse(source_ref)
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


def build_citation_url(
    source_ref: str,
    base_url: Optional[str] = None,
    ref_path: Optional[str] = None,
) -> Optional[str]:
    base = base_url or os.getenv("CITATION_BASE_URL") or os.getenv("DOCS_BASE_URL")
    if not base:
        return None
    path = ref_path or os.getenv("CITATION_REF_PATH", "/r/doc")
    base = base.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    encoded_ref = quote(source_ref, safe="")
    return f"{base}{path}?ref={encoded_ref}"


def _get_minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    secure = os.getenv("MINIO_SECURE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    if not access_key or not secret_key:
        raise RuntimeError("MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required for presigned URLs")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def resolve_link(
    source_ref: Optional[str] = None,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    version_id: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    page: Optional[int] = None,
    mode: Optional[str] = None,
) -> dict:
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
        return {"source_ref": source_ref, "url": url, "mode": resolved_mode}
    if resolved_mode == "cdn":
        base = os.getenv("CDN_BASE_URL", "").rstrip("/")
        if not base:
            raise RuntimeError("CDN_BASE_URL is required for CDN link resolution")
        safe_key = quote(key.lstrip("/"), safe="/")
        return {"source_ref": source_ref, "url": f"{base}/{safe_key}", "mode": resolved_mode}

    url = build_citation_url(source_ref)
    if not url:
        raise RuntimeError("CITATION_BASE_URL (or DOCS_BASE_URL) is required for portal links")
    return {"source_ref": source_ref, "url": url, "mode": "portal"}
