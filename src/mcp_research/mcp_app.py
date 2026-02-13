import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient

from mcp_research import hybrid_search
from mcp_research.link_resolver import build_citation_url, build_source_ref, resolve_link
from mcp_research.runtime_utils import decode_redis_value as _decode_redis_value, load_dotenv
from mcp_research.schema_v2 import (
    read_v2_doc_chunks,
    read_v2_doc_partitions,
    read_v2_source_doc_hash,
    redis_schema_read_mode,
    search_link_mode,
    search_redis_enrichment_mode,
    should_fallback_v1,
    should_read_v2,
)

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_URL = os.getenv("QDRANT_URL") or f"http://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "unstructured")
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
CITATION_BASE_URL = os.getenv("CITATION_BASE_URL") or os.getenv("DOCS_BASE_URL", "")
CITATION_REF_PATH = os.getenv("CITATION_REF_PATH", "/r/doc")
BIBTEX_REDIS_PREFIX = os.getenv("BIBTEX_REDIS_PREFIX", "bibtex").strip() or "bibtex"

_qdrant_client: QdrantClient | None = None
_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None
_redis_client = None


def _get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def _get_dense_model() -> TextEmbedding:
    """Return a cached dense embedding model."""
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding(model_name=DENSE_MODEL)
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    """Return a cached sparse embedding model."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def _get_models() -> tuple[TextEmbedding, SparseTextEmbedding]:
    """Return cached dense and sparse embedding models."""
    return _get_dense_model(), _get_sparse_model()


def _get_redis_client():
    """Return a cached Redis client when configured."""
    global _redis_client
    if not REDIS_URL:
        return None
    if redis is None:
        logger.warning("REDIS_URL set but redis package is missing; skipping Redis")
        return None
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL)
    return _redis_client


def _default_collection_key() -> str:
    """Return the Redis key used for storing the default collection name."""
    return f"{REDIS_PREFIX}:qdrant:default_collection"


def _redis_key(prefix: str, doc_id: str, suffix: str) -> str:
    """Build a Redis key for document metadata."""
    return f"{prefix}:pdf:{doc_id}:{suffix}"


def _source_key(prefix: str, source: str) -> str:
    """Build a Redis key for a source-to-document mapping."""
    return f"{prefix}:pdf:source:{source}"


def _get_default_collection() -> str | None:
    """Fetch the default collection name from Redis, if set."""
    client = _get_redis_client()
    if not client:
        return None
    return _decode_redis_value(client.get(_default_collection_key()))


def _pages_to_range(pages: List[int]) -> tuple[Optional[int], Optional[int]]:
    """Convert a list of page numbers to a min/max range."""
    if not pages:
        return None, None
    return min(pages), max(pages)


def _source_ref_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    """Construct a source_ref from payload fields or existing source_ref."""
    source_ref = payload.get("source_ref")
    if source_ref:
        return source_ref
    bucket = payload.get("bucket")
    key = payload.get("key")
    if not bucket or not key:
        return None
    page_start = payload.get("page_start")
    page_end = payload.get("page_end")
    if page_start is None and page_end is None:
        page_start, page_end = _pages_to_range(payload.get("pages", []))
    return build_source_ref(
        bucket=bucket,
        key=key,
        page_start=page_start,
        page_end=page_end,
        version_id=payload.get("version_id"),
    )


def _coerce_qdrant_offset(offset: str | None):
    """Coerce a Qdrant scroll offset to int when possible."""
    if offset is None:
        return None
    if offset.isdigit():
        return int(offset)
    return offset


def _file_identity(payload: Dict[str, Any]) -> Optional[tuple[str, Dict[str, Any]]]:
    """Derive a stable identity key and metadata from a Qdrant payload."""
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


def _coerce_int(value: Any) -> int | None:
    """Coerce a payload value into an int when possible."""
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


def _split_source(source: str | None) -> Tuple[str | None, str | None]:
    """Split source ('bucket/key') into bucket + key."""
    if not source or "/" not in source:
        return None, None
    bucket, key = source.split("/", 1)
    if not bucket or not key:
        return None, None
    return bucket, key


def _sorted_pages(payload: Dict[str, Any]) -> List[int]:
    """Return sorted, unique numeric pages from payload['pages'] when present."""
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []
    values = [_coerce_int(value) for value in pages]
    return sorted({value for value in values if value is not None})


def _payload_page_range(payload: Dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract a best-effort page_start/page_end range from a payload."""
    page_start = _coerce_int(payload.get("page_start"))
    page_end = _coerce_int(payload.get("page_end"))

    pages = _sorted_pages(payload)
    if page_start is None and pages:
        page_start = pages[0]
    if page_end is None and pages:
        page_end = pages[-1]

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        if page_start is None:
            page_start = _coerce_int(metadata.get("page_start"))
            if page_start is None:
                page_start = _coerce_int(metadata.get("page_number"))
            if page_start is None:
                page_start = _coerce_int(metadata.get("page_num"))
        if page_end is None:
            page_end = _coerce_int(metadata.get("page_end"))
            if page_end is None:
                page_end = _coerce_int(metadata.get("page_number"))
            if page_end is None:
                page_end = _coerce_int(metadata.get("page_num"))

    if page_start is None and page_end is not None:
        page_start = page_end
    if page_end is None and page_start is not None:
        page_end = page_start

    return page_start, page_end


def _partition_label(page_start: int | None, page_end: int | None) -> str:
    """Build a user-friendly partition label from a page range."""
    if page_start is None:
        return "Unknown pages"
    if page_end is None or page_end == page_start:
        return f"Page {page_start}"
    return f"Pages {page_start}-{page_end}"


def _extract_chunk_text(payload: Dict[str, Any]) -> str:
    """Extract chunk text from common payload fields."""
    for key in ("text", "chunk_text", "content", "chunk", "page_content", "raw_text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _source_and_doc_identity(payload: Dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Return (document_id, bucket, key) from payload fields."""
    document_id = payload.get("document_id")
    bucket = payload.get("bucket")
    key = payload.get("key")
    if bucket and key:
        return document_id, bucket, key

    source_bucket, source_key = _split_source(payload.get("source"))
    if source_bucket and source_key:
        return document_id, source_bucket, source_key
    return document_id, bucket, key


def _doc_meta(redis_client: Any, prefix: str, doc_id: str) -> Dict[str, str]:
    """Load document metadata hash from Redis when available."""
    meta_key = _redis_key(prefix, doc_id, "meta")
    if not hasattr(redis_client, "hgetall"):
        return {}
    raw_meta = redis_client.hgetall(meta_key) or {}
    normalized: Dict[str, str] = {}
    for raw_key, raw_value in raw_meta.items():
        key = _decode_redis_value(raw_key)
        value = _decode_redis_value(raw_value)
        if key is None or value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def _safe_json_list(raw_value: Any, *, field_name: str) -> List[Any]:
    """Parse a Redis JSON list field, raising a clear error on invalid JSON."""
    raw = _decode_redis_value(raw_value)
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON payload for {field_name}") from exc
    return payload if isinstance(payload, list) else []


def _resolve_doc_ids(
    redis_client: Any,
    document_id: str | None = None,
    bucket: str | None = None,
    key: str | None = None,
) -> tuple[List[str], str | None]:
    """Resolve one or more document IDs from explicit id or bucket/key source mapping."""
    if document_id:
        return [document_id], None
    if not (bucket and key):
        raise ValueError("document_id or bucket/key is required")

    source = f"{bucket}/{key}"
    read_mode = redis_schema_read_mode()
    if should_read_v2(read_mode):
        v2_doc = read_v2_source_doc_hash(
            redis_client=redis_client,
            prefix=REDIS_PREFIX,
            bucket=bucket,
            key=key,
            version_id=None,
        )
        if v2_doc:
            return [v2_doc], source
        if not should_fallback_v1(read_mode):
            return [], source

    source_key = _source_key(REDIS_PREFIX, source)
    doc_ids: List[str] = []

    raw_members = redis_client.smembers(source_key) if hasattr(redis_client, "smembers") else None
    for entry in raw_members or []:
        decoded = _decode_redis_value(entry)
        if decoded:
            doc_ids.append(str(decoded))

    if not doc_ids and hasattr(redis_client, "get"):
        mapped = _decode_redis_value(redis_client.get(source_key))
        if mapped:
            doc_ids.append(str(mapped))

    if not doc_ids:
        return [], source
    return sorted({doc_id for doc_id in doc_ids if doc_id}), source


def _load_document_payloads(
    redis_client: Any,
    doc_ids: List[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    """Load and merge Redis partitions + chunks payloads for one or more doc ids."""
    chunks: List[Dict[str, Any]] = []
    partitions: List[Dict[str, Any]] = []
    chunks_keys: List[str] = []
    partitions_keys: List[str] = []
    read_mode = redis_schema_read_mode()

    for doc_id in doc_ids:
        used_v2 = False
        if should_read_v2(read_mode):
            v2_chunks = read_v2_doc_chunks(redis_client, REDIS_PREFIX, doc_id)
            v2_partitions = read_v2_doc_partitions(redis_client, REDIS_PREFIX, doc_id)
            if v2_chunks or v2_partitions:
                used_v2 = True
                for entry in v2_partitions:
                    if isinstance(entry, dict):
                        partitions.append(dict(entry))
                for entry in v2_chunks:
                    if isinstance(entry, dict):
                        chunks.append(dict(entry))
                if v2_chunks:
                    chunks_keys.append(f"{REDIS_PREFIX}:v2:doc:{doc_id}:chunk_hashes")
                if v2_partitions:
                    partitions_keys.append(f"{REDIS_PREFIX}:v2:doc:{doc_id}:partition_hashes")
        if used_v2 and not should_fallback_v1(read_mode):
            continue

        meta = _doc_meta(redis_client, REDIS_PREFIX, doc_id)
        partitions_key = str(meta.get("partitions_key") or _redis_key(REDIS_PREFIX, doc_id, "partitions"))
        chunks_key = str(meta.get("chunks_key") or _redis_key(REDIS_PREFIX, doc_id, "chunks"))

        raw_partitions = redis_client.get(partitions_key)
        raw_chunks = redis_client.get(chunks_key)
        try:
            partition_entries = _safe_json_list(raw_partitions, field_name=partitions_key)
        except RuntimeError:
            partition_entries = []
        chunk_entries = _safe_json_list(raw_chunks, field_name=chunks_key)

        for entry in partition_entries:
            if isinstance(entry, dict):
                partitions.append(dict(entry))
        for entry in chunk_entries:
            if isinstance(entry, dict):
                chunks.append(dict(entry))

        if partition_entries:
            partitions_keys.append(partitions_key)
        if chunk_entries:
            chunks_keys.append(chunks_key)

    return chunks, partitions, chunks_keys, partitions_keys


def _fetch_document_bundle(
    redis_client: Any,
    document_id: str | None = None,
    bucket: str | None = None,
    key: str | None = None,
) -> Dict[str, Any]:
    """Fetch Redis-backed chunk/partition payloads for one document identity."""
    doc_ids, source = _resolve_doc_ids(
        redis_client=redis_client,
        document_id=document_id,
        bucket=bucket,
        key=key,
    )
    if not doc_ids:
        return {
            "document_id": document_id,
            "document_ids": [],
            "source": source,
            "found": False,
            "count": 0,
            "chunks": [],
            "partitions": [],
            "chunks_keys": [],
            "partitions_keys": [],
        }

    chunks, partitions, chunks_keys, partitions_keys = _load_document_payloads(redis_client, doc_ids)
    citation_cache: Dict[str, str | None] = {}
    normalized_chunks = [_enrich_chunk_payload(chunk, citation_cache) for chunk in chunks]

    return {
        "document_id": document_id if len(doc_ids) == 1 else None,
        "document_ids": doc_ids,
        "source": source,
        "found": bool(normalized_chunks),
        "count": len(normalized_chunks),
        "chunks": normalized_chunks,
        "partitions": partitions,
        "chunks_keys": chunks_keys,
        "partitions_keys": partitions_keys,
    }


def _bibtex_file_key(prefix: str, bucket: str, object_name: str) -> str:
    """Build the Redis key used by the BibTeX UI for file metadata."""
    return f"{prefix}:file:{bucket}/{object_name}"


def _load_bibtex_metadata(bucket: str | None, key: str | None) -> Dict[str, Any] | None:
    """Load BibTeX metadata from Redis for a bucket/key pair."""
    if not bucket or not key:
        return None
    redis_client = _get_redis_client()
    if not redis_client:
        return None
    raw = _decode_redis_value(redis_client.get(_bibtex_file_key(BIBTEX_REDIS_PREFIX, bucket, key)))
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _citation_key_from_payload(payload: Dict[str, Any]) -> str | None:
    """Extract citation key from payload or linked BibTeX metadata."""
    for key in ("citation_key", "citationKey"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    bucket = payload.get("bucket")
    object_key = payload.get("key")
    bibtex = _load_bibtex_metadata(bucket, object_key)
    if not bibtex:
        return None
    for key in ("citationKey", "citation_key"):
        value = bibtex.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _citation_key_for_object(
    bucket: str | None,
    key: str | None,
    cache: Dict[str, str | None],
) -> str | None:
    """Resolve citation key for a bucket/key pair with a small per-call cache."""
    if not bucket or not key:
        return None
    cache_key = f"{bucket}/{key}"
    if cache_key in cache:
        return cache[cache_key]
    metadata = _load_bibtex_metadata(bucket, key)
    citation_key = None
    if metadata:
        for field in ("citationKey", "citation_key"):
            value = metadata.get(field)
            if isinstance(value, str) and value.strip():
                citation_key = value.strip()
                break
    cache[cache_key] = citation_key
    return citation_key


def _enrich_chunk_payload(
    chunk: Dict[str, Any],
    citation_cache: Dict[str, str | None],
) -> Dict[str, Any]:
    """Attach citation and source link fields to a chunk payload."""
    source_ref = _source_ref_from_payload(chunk)
    if source_ref:
        chunk.setdefault("source_ref", source_ref)
        chunk.setdefault(
            "citation_url",
            build_citation_url(source_ref, CITATION_BASE_URL, CITATION_REF_PATH),
        )
    citation_key = _citation_key_from_payload(chunk)
    if citation_key is None:
        citation_key = _citation_key_for_object(chunk.get("bucket"), chunk.get("key"), citation_cache)
    if citation_key:
        chunk.setdefault("citation_key", citation_key)
    return chunk


def _chunk_summary(payload: Dict[str, Any], point_id: str | None = None) -> Dict[str, Any]:
    """Build a compact chunk summary object used by chunk-level fetch helpers."""
    page_start, page_end = _payload_page_range(payload)
    summary = {
        "point_id": point_id,
        "document_id": payload.get("document_id"),
        "bucket": payload.get("bucket"),
        "key": payload.get("key"),
        "source": payload.get("source"),
        "chunk_index": _coerce_int(payload.get("chunk_index")),
        "page_start": page_start,
        "page_end": page_end,
        "pages": _sorted_pages(payload),
        "text": _extract_chunk_text(payload),
    }
    source_ref = _source_ref_from_payload(payload)
    if source_ref:
        summary["source_ref"] = source_ref
        summary["citation_url"] = build_citation_url(source_ref, CITATION_BASE_URL, CITATION_REF_PATH)
    citation_key = _citation_key_from_payload(payload)
    if citation_key:
        summary["citation_key"] = citation_key
    return summary


def _partition_details_from_bundle(
    payload: Dict[str, Any],
    bundle: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Build a partition summary for one payload against a loaded document bundle."""
    target_start, target_end = _payload_page_range(payload)
    target_chunk_index = _coerce_int(payload.get("chunk_index"))
    matching_chunks: List[Dict[str, Any]] = []
    for chunk in bundle.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        entry = dict(chunk)
        start, end = _payload_page_range(entry)
        if target_start is not None or target_end is not None:
            if (start, end) == (target_start, target_end):
                matching_chunks.append(entry)
        elif target_chunk_index is not None:
            chunk_index = _coerce_int(entry.get("chunk_index"))
            if chunk_index == target_chunk_index:
                matching_chunks.append(entry)

    matching_partitions: List[Dict[str, Any]] = []
    for partition in bundle.get("partitions", []):
        if not isinstance(partition, dict):
            continue
        entry = dict(partition)
        start, end = _payload_page_range(entry)
        if target_start is not None or target_end is not None:
            if (start, end) == (target_start, target_end):
                matching_partitions.append(entry)

    selected_partition = matching_partitions[0] if matching_partitions else None
    if not matching_chunks and not selected_partition:
        return None
    return {
        "label": _partition_label(target_start, target_end),
        "page_start": target_start,
        "page_end": target_end,
        "chunk_count": len(matching_chunks),
        "partition_payload": selected_partition,
    }


def _retrieve_chunk_payload(id: str, collection: str | None = None) -> tuple[str, Dict[str, Any] | None]:
    """Retrieve chunk payload from Qdrant by id and collection selection."""
    client = _get_qdrant_client()
    collection_name = collection or _get_default_collection() or QDRANT_COLLECTION
    points = client.retrieve(
        collection_name=collection_name,
        ids=[id],
        with_payload=True,
    )
    if not points:
        return collection_name, None
    return collection_name, points[0].payload or {}


def _to_float_list(values: Any) -> List[float]:
    """Convert embedding values to a plain Python float list."""
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return [float(value) for value in values]


def _normalize_retrieval_mode(value: str | None) -> str:
    """Normalize and validate retrieval mode."""
    mode = str(value or "hybrid").strip().lower()
    if mode not in {"hybrid", "cosine"}:
        raise ValueError("retrieval_mode must be one of: hybrid, cosine")
    return mode


def _cosine_search(
    client: QdrantClient,
    collection_name: str,
    dense_model: TextEmbedding,
    query_text: str,
    top_k: int,
):
    """Run dense-only cosine similarity search against the named dense vector."""
    query_vector = _to_float_list(next(iter(dense_model.embed([query_text]))))
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="dense",
        limit=top_k,
        with_payload=True,
    )

mcp = FastMCP(name="Local MCP Server", auth=None)


@mcp.tool
def ping() -> str:
    """Simple health-check."""
    return "pong"

@mcp.tool
def list_collections() -> Dict[str, List[str]]:
    """List Qdrant collections."""
    client = _get_qdrant_client()
    collections = client.get_collections()
    names = [entry.name for entry in collections.collections]
    return {"collections": names}


@mcp.tool
def set_default_collection(name: str) -> Dict[str, str]:
    """Set the default Qdrant collection name in Redis."""
    client = _get_qdrant_client()
    if not client.collection_exists(name):
        raise ValueError(f"Collection not found: {name}")
    redis_client = _get_redis_client()
    if not redis_client:
        raise RuntimeError("REDIS_URL is required to store the default collection")
    redis_client.set(_default_collection_key(), name)
    return {"default_collection": name}


@mcp.tool
def list_collection_files(
    collection: str | None = None,
    limit: int = 200,
    batch_size: int = 256,
    offset: str | None = None,
) -> Dict[str, Any]:
    """
    List unique files in a Qdrant collection by scanning chunk payloads.
    """
    client = _get_qdrant_client()
    collection_name = collection or _get_default_collection() or QDRANT_COLLECTION
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
            if identity not in files:
                info["chunks"] = 0
                files[identity] = info
            files[identity]["chunks"] += 1
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
        "files": list(files.values()),
    }


@mcp.tool
def search(
    query: str,
    top_k: int = 5,
    prefetch_k: int = 40,
    collection: str | None = None,
    retrieval_mode: str = "hybrid",
    include_partition: bool | None = None,
    include_document: bool | None = None,
) -> Dict[str, Any]:
    """
    Search indexed chunks via hybrid (dense+sparse) retrieval
    or dense-only cosine similarity retrieval.
    Returns results with ids that can be passed to fetch().
    """
    client = _get_qdrant_client()
    mode = _normalize_retrieval_mode(retrieval_mode)
    dense_model = _get_dense_model()

    collection_name = collection or _get_default_collection() or QDRANT_COLLECTION
    enrichment_mode = search_redis_enrichment_mode()
    include_partition = (
        include_partition
        if include_partition is not None
        else enrichment_mode in {"partition", "both"}
    )
    include_document = (
        include_document
        if include_document is not None
        else enrichment_mode in {"document", "both"}
    )
    if mode == "cosine":
        response = _cosine_search(
            client=client,
            collection_name=collection_name,
            dense_model=dense_model,
            query_text=query,
            top_k=top_k,
        )
    else:
        sparse_model = _get_sparse_model()
        response = hybrid_search.hybrid_search(
            client=client,
            collection_name=collection_name,
            dense_model=dense_model,
            sparse_model=sparse_model,
            query_text=query,
            top_k=top_k,
            prefetch_k=prefetch_k,
        )

    results: List[Dict[str, Any]] = []
    redis_client = _get_redis_client()
    bundle_cache: Dict[str, Dict[str, Any]] = {}
    for point in response.points:
        payload = point.payload or {}
        source_ref = _source_ref_from_payload(payload)
        citation_url = (
            build_citation_url(source_ref, CITATION_BASE_URL, CITATION_REF_PATH)
            if source_ref
            else None
        )
        citation_key = _citation_key_from_payload(payload)
        citation_url_doc_start = None
        citation_url_match = None
        if source_ref and search_link_mode() == "dual":
            citation_url_doc_start = resolve_link(source_ref=source_ref, page=1).get("url")
            page_start, page_end = _payload_page_range(payload)
            citation_url_match = resolve_link(
                source_ref=source_ref,
                page_start=page_start,
                page_end=page_end,
                highlight=_extract_chunk_text(payload)[:240],
            ).get("url")

        partition_data = None
        document_data = None
        if redis_client and (include_partition or include_document):
            document_id, bucket, key = _source_and_doc_identity(payload)
            cache_key = f"{document_id or ''}|{bucket or ''}|{key or ''}"
            if cache_key not in bundle_cache:
                bundle_cache[cache_key] = _fetch_document_bundle(
                    redis_client=redis_client,
                    document_id=document_id,
                    bucket=bucket,
                    key=key,
                )
            bundle = bundle_cache[cache_key]
            if include_partition:
                partition_data = _partition_details_from_bundle(payload, bundle)
            if include_document:
                document_data = {
                    "document_ids": bundle.get("document_ids", []),
                    "count": bundle.get("count", 0),
                }

        results.append(
            {
                "id": str(point.id),
                "score": point.score,
                "document_id": payload.get("document_id"),
                "source": payload.get("source"),
                "source_ref": source_ref,
                "citation_url": citation_url,
                "citation_key": citation_key,
                "bucket": payload.get("bucket"),
                "key": payload.get("key"),
                "version_id": payload.get("version_id"),
                "chunk_index": _coerce_int(payload.get("chunk_index")),
                "pages": payload.get("pages", []),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "text": payload.get("text", ""),
                "partition": partition_data,
                "document": document_data,
                "citation_url_doc_start": citation_url_doc_start,
                "citation_url_match": citation_url_match,
            }
        )

    return {
        "retrieval_mode": mode,
        "include_partition": include_partition,
        "include_document": include_document,
        "results": results,
    }


@mcp.tool
def fetch(id: str, collection: str | None = None) -> Dict[str, Any]:
    """
    Fetch a single chunk by id for deep retrieval.
    """
    collection_name, payload = _retrieve_chunk_payload(id=id, collection=collection)
    if payload is None:
        return {"id": id, "found": False}
    source_ref = _source_ref_from_payload(payload)
    citation_url = (
        build_citation_url(source_ref, CITATION_BASE_URL, CITATION_REF_PATH)
        if source_ref
        else None
    )
    citation_key = _citation_key_from_payload(payload)
    return {
        "id": str(id),
        "found": True,
        "collection": collection_name,
        "document_id": payload.get("document_id"),
        "source": payload.get("source"),
        "source_ref": source_ref,
        "citation_url": citation_url,
        "citation_key": citation_key,
        "bucket": payload.get("bucket"),
        "key": payload.get("key"),
        "version_id": payload.get("version_id"),
        "chunk_index": _coerce_int(payload.get("chunk_index")),
        "pages": payload.get("pages", []),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "text": payload.get("text", ""),
    }


@mcp.tool
def resolve_citation(
    source_ref: str | None = None,
    bucket: str | None = None,
    key: str | None = None,
    version_id: str | None = None,
    page: int | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    highlight: str | None = None,
    mode: str | None = None,
) -> Dict[str, Any]:
    """Resolve a citation to a stable portal URL or a presigned MinIO URL."""
    return resolve_link(
        source_ref=source_ref,
        bucket=bucket,
        key=key,
        version_id=version_id,
        page=page,
        page_start=page_start,
        page_end=page_end,
        highlight=highlight,
        mode=mode,
    )


@mcp.tool
def fetch_document_chunks(
    document_id: str | None = None,
    bucket: str | None = None,
    key: str | None = None,
) -> Dict[str, Any]:
    """
    Fetch all chunk payloads for a document id or bucket/key stored in Redis.
    """
    redis_client = _get_redis_client()
    if not redis_client:
        raise RuntimeError("REDIS_URL is required to fetch document chunks")
    bundle = _fetch_document_bundle(
        redis_client=redis_client,
        document_id=document_id,
        bucket=bucket,
        key=key,
    )
    if not bundle.get("found"):
        return {
            "document_id": bundle.get("document_id"),
            "document_ids": bundle.get("document_ids", []),
            "source": bundle.get("source"),
            "found": False,
            "chunks": [],
        }
    chunks_keys = bundle.get("chunks_keys", [])
    return {
        "document_id": bundle.get("document_id"),
        "document_ids": bundle.get("document_ids", []),
        "source": bundle.get("source"),
        "found": True,
        "chunks_key": chunks_keys[0] if chunks_keys else None,
        "chunks_keys": chunks_keys,
        "count": bundle.get("count", 0),
        "chunks": bundle.get("chunks", []),
    }


@mcp.tool
def fetch_chunk_document(id: str, collection: str | None = None) -> Dict[str, Any]:
    """
    Fetch the full Redis chunk payload for the document that contains this chunk id.
    """
    clean_id = str(id).strip()
    if not clean_id:
        raise ValueError("id is required")

    collection_name, payload = _retrieve_chunk_payload(id=clean_id, collection=collection)
    if payload is None:
        return {"id": clean_id, "found": False}

    document_id, bucket, key = _source_and_doc_identity(payload)
    redis_client = _get_redis_client()
    if not redis_client:
        raise RuntimeError("REDIS_URL is required to fetch document chunks")

    bundle = _fetch_document_bundle(
        redis_client=redis_client,
        document_id=document_id,
        bucket=bucket,
        key=key,
    )
    chunk = _chunk_summary(payload, point_id=clean_id)
    return {
        "id": clean_id,
        "collection": collection_name,
        "found": bool(bundle.get("found")),
        "chunk": chunk,
        "document_id": document_id,
        "document_ids": bundle.get("document_ids", []),
        "source": bundle.get("source"),
        "chunks_key": (bundle.get("chunks_keys", []) or [None])[0],
        "chunks_keys": bundle.get("chunks_keys", []),
        "count": bundle.get("count", 0),
        "chunks": bundle.get("chunks", []),
    }


@mcp.tool
def fetch_chunk_partition(id: str, collection: str | None = None) -> Dict[str, Any]:
    """
    Fetch the partition (page range) and partition chunks that contain this chunk id.
    """
    clean_id = str(id).strip()
    if not clean_id:
        raise ValueError("id is required")

    collection_name, payload = _retrieve_chunk_payload(id=clean_id, collection=collection)
    if payload is None:
        return {"id": clean_id, "found": False}

    document_id, bucket, key = _source_and_doc_identity(payload)
    redis_client = _get_redis_client()
    if not redis_client:
        raise RuntimeError("REDIS_URL is required to fetch partition chunks")

    bundle = _fetch_document_bundle(
        redis_client=redis_client,
        document_id=document_id,
        bucket=bucket,
        key=key,
    )
    if not bundle.get("found"):
        return {
            "id": clean_id,
            "collection": collection_name,
            "found": False,
            "chunk": _chunk_summary(payload, point_id=clean_id),
            "partition": None,
            "chunks": [],
        }

    target_start, target_end = _payload_page_range(payload)
    target_chunk_index = _coerce_int(payload.get("chunk_index"))

    matching_chunks: List[Dict[str, Any]] = []
    for chunk in bundle.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        entry = dict(chunk)
        start, end = _payload_page_range(entry)
        if target_start is not None or target_end is not None:
            if (start, end) == (target_start, target_end):
                matching_chunks.append(entry)
        elif target_chunk_index is not None:
            chunk_index = _coerce_int(entry.get("chunk_index"))
            if chunk_index == target_chunk_index:
                matching_chunks.append(entry)

    matching_partitions: List[Dict[str, Any]] = []
    for partition in bundle.get("partitions", []):
        if not isinstance(partition, dict):
            continue
        entry = dict(partition)
        start, end = _payload_page_range(entry)
        if target_start is not None or target_end is not None:
            if (start, end) == (target_start, target_end):
                matching_partitions.append(entry)

    selected_partition = matching_partitions[0] if matching_partitions else None
    partition_payload = {
        "label": _partition_label(target_start, target_end),
        "page_start": target_start,
        "page_end": target_end,
        "chunk_count": len(matching_chunks),
        "partitions_key": (bundle.get("partitions_keys", []) or [None])[0],
        "partition_payload": selected_partition,
    }

    return {
        "id": clean_id,
        "collection": collection_name,
        "found": bool(matching_chunks or selected_partition),
        "chunk": _chunk_summary(payload, point_id=clean_id),
        "document_id": document_id,
        "document_ids": bundle.get("document_ids", []),
        "source": bundle.get("source"),
        "partition": partition_payload,
        "count": len(matching_chunks),
        "chunks": matching_chunks,
    }


@mcp.tool
def fetch_chunk_bibtex(id: str, collection: str | None = None) -> Dict[str, Any]:
    """
    Fetch BibTeX metadata for the object that contains this chunk id.
    """
    clean_id = str(id).strip()
    if not clean_id:
        raise ValueError("id is required")

    collection_name, payload = _retrieve_chunk_payload(id=clean_id, collection=collection)
    if payload is None:
        return {"id": clean_id, "found": False}

    _, bucket, key = _source_and_doc_identity(payload)
    metadata = _load_bibtex_metadata(bucket=bucket, key=key)
    citation_key = _citation_key_from_payload(payload)
    if citation_key is None and metadata:
        citation_key = str(metadata.get("citationKey") or metadata.get("citation_key") or "").strip() or None

    return {
        "id": clean_id,
        "collection": collection_name,
        "found": bool(metadata),
        "bucket": bucket,
        "key": key,
        "citation_key": citation_key,
        "metadata": metadata or {},
    }


# FastMCP decorators may replace function objects with tool wrappers.
# Keep module-level callables for unit tests and internal direct invocation.
for _tool_name in (
    "list_collections",
    "set_default_collection",
    "list_collection_files",
    "search",
    "fetch",
    "resolve_citation",
    "fetch_document_chunks",
    "fetch_chunk_document",
    "fetch_chunk_partition",
    "fetch_chunk_bibtex",
):
    _tool_obj = globals().get(_tool_name)
    if _tool_obj is not None and hasattr(_tool_obj, "fn"):
        globals()[_tool_name] = _tool_obj.fn


def main() -> None:
    """Run the FastMCP server with HTTP transport."""
    # Remote deployment uses HTTP transport; default MCP path is /mcp
    mcp.run(transport="http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
