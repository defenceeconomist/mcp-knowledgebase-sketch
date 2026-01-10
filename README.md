# GitHub OAuth MCP Server

FastMCP server that authenticates users via GitHub OAuth and exposes two simple tools (`whoami`, `ping`). It can run locally or via Docker, and includes an optional Cloudflare Tunnel sidecar for exposing the service publicly.

## Requirements
- Python 3.12+
- Docker + Docker Compose (optional, for containerized runs)
- GitHub OAuth app credentials and a public callback URL (tunnel or other)

## Environment Variables
Set these in your shell or a local `.env` (keep secrets out of version control):
- `FASTMCP_SERVER_AUTH=fastmcp.server.auth.providers.github.GitHubProvider`
- `FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID` – OAuth app client ID
- `FASTMCP_SERVER_AUTH_GITHUB_CLIENT_SECRET` – OAuth app client secret
- `FASTMCP_SERVER_AUTH_GITHUB_BASE_URL` – Public base URL where `/auth` is reachable (e.g. your tunnel host)
- `CF_TUNNEL_TOKEN` – Only needed when running the Cloudflare Tunnel sidecar

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
The server listens on `http://0.0.0.0:8000`. OAuth callbacks resolve via `FASTMCP_SERVER_AUTH_GITHUB_BASE_URL`.

## Run with Docker
Build and start the stack (FastMCP server + optional Cloudflare Tunnel):
```bash
docker compose up --build
```
- `mcp` service: serves HTTP MCP on port 8000.
- `cloudflared` service: starts a tunnel using `CF_TUNNEL_TOKEN` so the OAuth callback is reachable from GitHub.

## Exposed Tools
- `whoami`: returns authenticated GitHub user info (`login`, `name`, `email`).
- `ping`: health check returning `"pong"`.

## Notes
- Configure your GitHub OAuth app callback to `<base_url>/auth/callback`, where `<base_url>` is `FASTMCP_SERVER_AUTH_GITHUB_BASE_URL`.
- Do not commit real OAuth secrets or tunnel tokens. Use a local `.env` only for development.

## PDF -> Qdrant ingestion
Use `ingest_pdfs.py` to extract text from PDFs with PyMuPDF, chunk it, embed with SentenceTransformers, and push vectors to the local Qdrant instance.

Environment (examples):
- `DATA_DIR` – directory containing PDFs when `PDF_PATH` is not set (default `data`).
- `PDF_PATH` – optional path to a single PDF or a directory to ingest.
- `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION` – defaults to the local compose stack (`qdrant`, `6333`, `pdf_chunks`).
- `EMBEDDING_MODEL` – SentenceTransformers model name (default `all-MiniLM-L6-v2`).
- `CHUNK_SIZE` / `CHUNK_OVERLAP` – chunking controls (defaults `1200` / `200` characters).

Run locally:
```bash
python ingest_pdfs.py              # ingests PDFs in ./data
PDF_PATH=./data/my.pdf python ingest_pdfs.py  # ingest a single file
```

Run inside Docker (uses mounted `./data`):
```bash
docker compose run --rm mcp python ingest_pdfs.py
```

The script will create the `pdf_chunks` collection (if missing) and upsert chunk payloads with page hints and text for retrieval.

### Using the Unstructured API (hybrid dense + sparse vectors with FastEmbed)
Use `ingest_unstructured.py` when you want Unstructured to handle PDF parsing/chunking and store both dense (`dense`) and sparse (`sparse`, BM25-style) vectors in Qdrant. FastEmbed powers both vector types; Qdrant stores them as named vectors for hybrid search.

Key environment:
- `DATA_DIR` – directory of PDFs when `PDF_PATH` is not set (default `data`).
- `PDF_PATH` – path to a single PDF **or** a directory (both accepted).
- `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION` – customize the target collection (default `pdf_chunks`).
- `DENSE_MODEL` / `SPARSE_MODEL` – FastEmbed model names (defaults: `BAAI/bge-small-en-v1.5` and `Qdrant/bm25`).
- `UNSTRUCTURED_API_KEY` (required) – API key for the Unstructured API.
- `UNSTRUCTURED_API_URL` – base URL (default `https://api.unstructured.io`).
- `UNSTRUCTURED_STRATEGY` – partition strategy (default `hi_res`).
- `UNSTRUCTURED_CHUNKING_STRATEGY` – chunking mode (`basic` by default, set to `none` to disable).
- `CHUNK_SIZE` / `CHUNK_OVERLAP` – forwarded to the Unstructured chunker as `max_characters` / `overlap`.
- `UNSTRUCTURED_LANGUAGES` – optional comma-separated language codes (e.g., `eng,spa`).

Run locally (directory ingest, default collection):
```bash
UNSTRUCTURED_API_KEY=your-key python ingest_unstructured.py
```

Single-file ingest:
```bash
UNSTRUCTURED_API_KEY=your-key PDF_PATH=./data/my.pdf python ingest_unstructured.py
```

Custom collection name:
```bash
UNSTRUCTURED_API_KEY=your-key QDRANT_COLLECTION=my_hybrid_chunks python ingest_unstructured.py
```

Run inside Docker (uses mounted `./data`; swap envs as needed):
```bash
UNSTRUCTURED_API_KEY=your-key docker compose run --rm mcp python ingest_unstructured.py
```

The Dockerfile uses a Python entrypoint, so you can swap commands as needed (server default remains `python mcp_app.py`).

## Hybrid search demo (local)
Use `hybrid_search.py` to run a small hybrid (dense + sparse) search against a local Qdrant instance. The script can create a demo collection, index sample docs, and then query with reciprocal rank fusion.

Prereqs:
- Qdrant running locally (e.g. `docker compose up qdrant`)
- Python deps installed: `pip install -r requirements.txt`

Environment (optional):
- `QDRANT_URL` – Qdrant URL (default `http://localhost:6333`)
- `QDRANT_COLLECTION` – collection name (default `hybrid_demo`)
- `DENSE_MODEL` – FastEmbed dense model name (default `BAAI/bge-small-en-v1.5`)
- `SPARSE_MODEL` – FastEmbed sparse model name (default `Qdrant/bm25`)

Run locally:
```bash
python hybrid_search.py "hybrid search in qdrant" --recreate
```

Custom URL/collection:
```bash
QDRANT_URL=http://localhost:6333 QDRANT_COLLECTION=my_hybrid_demo \
  python hybrid_search.py "vector search" --top-k 5 --prefetch-k 40
```

## Hybrid search examples
Two runnable scripts live in `examples/`:

Query an existing collection:
```bash
python examples/search_existing.py "your query" my_collection
```

Seed demo docs (if empty) and query:
```bash
python examples/search_demo.py "your query" demo_collection
```

Recreate the demo collection before indexing:
```bash
python examples/search_demo.py "your query" demo_collection --recreate
```
