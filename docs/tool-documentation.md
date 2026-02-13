# Tool Documentation

This document focuses on:
1. Data ingest process
2. Data schema
3. Exposed MCP tools

## 1) Data Ingest Process

### A. Local PDF ingest (batch/manual)

Main modules:
- `src/mcp_research/ingest_unstructured.py`
- `src/mcp_research/upsert_chunks.py`

Flow:
1. Read PDF(s) from `PDF_PATH` or `DATA_DIR`.
2. Call Unstructured API (`UNSTRUCTURED_API_KEY` required) to partition/chunk.
3. Build chunk payloads with:
   - `document_id` (SHA-256 of PDF bytes)
   - page metadata (`pages`, `page_start`, `page_end`)
   - source metadata (`source`, `bucket`, `key`, `source_ref`)
4. Optionally write partitions/chunks JSON to disk (`STORE_PARTITIONS_DISK`, `STORE_CHUNKS_DISK`).
5. Write metadata/chunks to Redis via `upload_to_redis()` (schema mode controlled by `REDIS_SCHEMA_WRITE_MODE`).
6. Upsert chunk vectors into Qdrant via `mcp_cli.py upsert-chunks` (dense + sparse vectors).

Typical commands:

```bash
python mcp_cli.py ingest-unstructured
python mcp_cli.py upsert-chunks --source redis --redis-url redis://localhost:6379/0
```

Notes:
- `ingest-unstructured` does not write to Qdrant directly.
- Qdrant indexing happens in `upsert-chunks`.

### B. MinIO event-driven ingest

Main module:
- `src/mcp_research/minio_ingest.py`

Flow:
1. Listen to MinIO notifications (`s3:ObjectCreated:*`, `s3:ObjectRemoved:*` by default).
2. For created/updated PDFs:
   - download object bytes from MinIO
   - derive `document_id` (SHA-256)
   - partition/chunk with Unstructured (if not already reusable from Redis)
   - write Redis metadata/chunks
   - upsert vectors into Qdrant collection named after bucket
3. For removed objects:
   - delete matching Qdrant points by `{bucket, key, version_id}`
   - clean Redis source/collection mappings

Celery mode:
- If `MINIO_ENQUEUE_CELERY=1`, listener enqueues:
  - `mcp_research.ingest_minio_object`
  - `mcp_research.delete_minio_object`
- Worker implementation: `src/mcp_research/ingestion_tasks.py`

### C. Backfill and recovery utilities

- `src/mcp_research/upload_data_to_redis.py`:
  upload partition/chunk JSON payloads into Redis.
- `src/mcp_research/ingest_missing_minio.py`:
  detect MinIO files missing in Redis and enqueue ingest tasks.

## 2) Data Schema

## 2.1 Core identity and hash fields

- `document_id`/`doc_hash`: SHA-256 of PDF bytes
- `source_id`: SHA-1 of `s3://{bucket}/{key}?version_id={version_id}`
- `partition_hash`: SHA-256 of normalized partition identity
- `chunk_hash`: SHA-256 of normalized chunk identity
- `source_ref`: `doc://{bucket}/{key}[?version_id=...][#page=...]`

## 2.2 Redis schema (v1)

Prefix default: `unstructured`.

Primary keys:
- `{prefix}:pdf:{doc_id}:meta` (hash)
- `{prefix}:pdf:{doc_id}:partitions` (JSON list)
- `{prefix}:pdf:{doc_id}:chunks` (JSON list)
- `{prefix}:pdf:{doc_id}:collections` (set)
- `{prefix}:pdf:hashes` (set of doc ids)
- `{prefix}:pdf:source:{bucket}/{key}` (set or scalar mapping to doc id)

## 2.3 Redis schema (v2)

Normalized keys:
- `{prefix}:v2:doc_hashes`
- `{prefix}:v2:doc:{doc_hash}:meta`
- `{prefix}:v2:doc:{doc_hash}:sources`
- `{prefix}:v2:doc:{doc_hash}:collections`
- `{prefix}:v2:source:{source_id}:meta`
- `{prefix}:v2:source:{source_id}:doc`
- `{prefix}:v2:doc:{doc_hash}:partition_hashes` (zset)
- `{prefix}:v2:partition:{partition_hash}` (JSON)
- `{prefix}:v2:partition:{partition_hash}:chunk_hashes` (zset)
- `{prefix}:v2:doc:{doc_hash}:chunk_hashes` (zset)
- `{prefix}:v2:chunk:{chunk_hash}` (JSON)

Schema switches:
- `REDIS_SCHEMA_WRITE_MODE`: `v1`, `dual`, `v2`
- `REDIS_SCHEMA_READ_MODE`: `v1`, `prefer_v2`, `v2`

## 2.4 Qdrant schema

Collection layout:
- named dense vector: `dense` (cosine distance)
- named sparse vector: `sparse` (IDF modifier)

Point payload (v1 fields):
- `document_id`, `source`, `chunk_index`, `pages`, `text`
- optional: `source_ref`, `bucket`, `key`, `version_id`, `page_start`, `page_end`

Point payload (v2 fields):
- `doc_hash`, `partition_hash`, `chunk_hash`
- optional: `source_id`

Payload/ID switches:
- `QDRANT_PAYLOAD_SCHEMA`: `v1`, `dual`, `v2`
- `QDRANT_POINT_ID_MODE`: `uuid` or `deterministic`

Deterministic ID mode uses UUIDv5 of:
- `{collection}|{doc_hash}|{chunk_hash}`

## 3) Exposed MCP Tools

Source:
- `src/mcp_research/mcp_app.py`

Tool list:

1. `ping()`
   - health check

2. `list_collections()`
   - returns available Qdrant collections

3. `set_default_collection(name)`
   - validates collection exists, stores default in Redis key:
   - `{REDIS_PREFIX}:qdrant:default_collection`

4. `list_collection_files(collection=None, limit=200, batch_size=256, offset=None)`
   - scans chunk payloads and returns unique files/doc identities

5. `search(query, top_k=5, prefetch_k=40, collection=None, retrieval_mode="hybrid", include_partition=None, include_document=None)`
   - retrieval modes:
   - `hybrid`: sparse+dense fusion
   - `cosine`: dense-only
   - returns chunk-level hits with source and citation metadata
   - optional Redis enrichment for partition/document context

6. `fetch(id, collection=None)`
   - fetch one chunk payload by Qdrant point id

7. `resolve_citation(source_ref=None, bucket=None, key=None, version_id=None, page=None, page_start=None, page_end=None, highlight=None, mode=None)`
   - resolves to portal/CDN/presigned URL depending on resolver mode

8. `fetch_document_chunks(document_id=None, bucket=None, key=None)`
   - returns all chunks for a document from Redis (supports v1/v2 read modes)

9. `fetch_chunk_document(id, collection=None)`
   - for one chunk id, returns the chunk plus full document chunk bundle from Redis

10. `fetch_chunk_partition(id, collection=None)`
    - for one chunk id, returns matching partition and all chunks in that page range

11. `fetch_chunk_bibtex(id, collection=None)`
    - returns BibTeX metadata for the chunk's source object from Redis

Operational requirements:
- Qdrant is required for search/fetch by point id.
- Redis is required for default collection storage and all document/partition enrichment tools.
- BibTeX lookup requires Redis keys under `BIBTEX_REDIS_PREFIX` (default `bibtex`).
