# Quick Start

## 1) Install
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2) Configure
Create `.env` and set:
- `QDRANT_URL` (default `http://localhost:6333`)
- `QDRANT_COLLECTION` (default `pdf_chunks`)
- `REDIS_URL` (default `redis://localhost:6379/0` for local)

(Optional) add:
- `UNSTRUCTURED_API_KEY`
- `CITATION_BASE_URL` (default `http://localhost:8080`)
- `CITATION_REF_PATH` (default `/r/doc`)

## 3) Run the MCP server
```bash
python mcp_app.py
```
Server listens on `http://localhost:8000` and serves MCP at `http://localhost:8000/mcp`.

## 3b) Run the dashboard (optional)
The dashboard shows Qdrant collections (used as buckets) and per-file metadata from Qdrant + Redis.

Run locally:
```bash
python -m mcp_research.dashboard_app
```
Open `http://localhost:8002`.

## 4) Run Qdrant (optional)
```bash
docker compose up qdrant
```

## 5) Run MinIO (optional UI for bucket uploads)
```bash
docker compose up minio
```
Open `http://localhost:9001` and sign in with `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` (defaults to `minioadmin`/`minioadmin`).

## 6) Hybrid search demo
```bash
python hybrid_search.py "hybrid search in qdrant" --recreate
```

## 7) Unstructured ingest + upsert (optional)
Ingest PDFs via Unstructured (stores partitions + chunks in Redis by default):
```bash
UNSTRUCTURED_API_KEY=your-key python ingest_unstructured.py
```

Upsert the generated chunks into Qdrant (from disk):
```bash
python upsert_chunks.py --chunks-dir ./data/chunks --collection test_collection
```

Upsert from Redis instead:
```bash
REDIS_URL=redis://localhost:6379/0 python upsert_chunks.py --source redis --collection test_collection
```

Run the same flow in Docker (uses mounted `./data`):
```bash
UNSTRUCTURED_API_KEY=your-key docker compose run --rm mcp python ingest_unstructured.py
docker compose run --rm mcp python upsert_chunks.py --chunks-dir /app/data/chunks --collection test_collection
```

If you run Redis locally (or via `docker compose up redis redisinsight`), partitions + chunks are mirrored into Redis and you can browse them in RedisInsight at `http://localhost:8001`. Set `STORE_PARTITIONS_DISK=1` and `STORE_CHUNKS_DISK=1` to also write JSON files to `./data`.

## 8) Batch-autofill BibTeX metadata (optional)
Populate BibTeX entries for PDFs from MinIO metadata + Redis first-page text + Crossref:
```bash
CROSSREF_MAILTO=you@example.com python bibtex_autofill.py --bucket your-bucket --limit 200
```

Preview without writing:
```bash
python bibtex_autofill.py --bucket your-bucket --dry-run
```
