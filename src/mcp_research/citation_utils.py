from __future__ import annotations

import os
from typing import Optional
from urllib.parse import quote, urlencode


def build_source_ref(
    bucket: str,
    key: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    version_id: Optional[str] = None,
) -> str:
    """Build a doc:// source reference for a bucket/key with optional page range."""
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
    base_url: Optional[str] = None,
    ref_path: Optional[str] = None,
) -> Optional[str]:
    """Build a citation portal URL that wraps a source reference."""
    base = (
        base_url
        or os.getenv("CITATION_BASE_URL")
        or os.getenv("DOCS_BASE_URL")
        or os.getenv("LINK_RESOLVER_BASE_URL")
        or "http://localhost:8080"
    )
    path = ref_path or os.getenv("CITATION_REF_PATH", "/r/doc")
    base = base.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    encoded_ref = quote(source_ref, safe="")
    return f"{base}{path}?ref={encoded_ref}"
