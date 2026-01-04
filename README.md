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

### Using the Unstructured API for partitioning + chunking
Use `ingest_unstructured.py` when you want Unstructured to handle PDF parsing and chunking before embedding and upserting into Qdrant.

Additional environment:
- `UNSTRUCTURED_API_KEY` (required) – API key for the Unstructured API.
- `UNSTRUCTURED_API_URL` – base URL (default `https://api.unstructured.io`).
- `UNSTRUCTURED_STRATEGY` – partition strategy (default `hi_res`).
- `UNSTRUCTURED_CHUNKING_STRATEGY` – chunking mode (`basic` by default, set to `none` to disable).
- `CHUNK_SIZE` / `CHUNK_OVERLAP` – forwarded to the Unstructured chunker as `max_characters` / `overlap`.
- `UNSTRUCTURED_LANGUAGES` – optional comma-separated language codes (e.g., `eng,spa`).

Run locally:
```bash
UNSTRUCTURED_API_KEY=your-key python ingest_unstructured.py
```
This uses the same Qdrant and SentenceTransformers defaults as `ingest_pdfs.py` but calls the Unstructured API to partition and chunk PDFs before embedding and upserting.

Run inside Docker (uses mounted `./data`):
```bash
UNSTRUCTURED_API_KEY=your-key docker compose run --rm mcp ingest_unstructured.py
```
The Dockerfile now uses a Python entrypoint so you can swap the command to run any script (server stays `python mcp_app.py` by default).

If you prefer to rely on `.env`, add `UNSTRUCTURED_API_KEY=...` there, rebuild, and run:
```bash
docker compose build
docker compose run --rm mcp ingest_unstructured.py
```

Try

```bash
# note the python is not strictly required in this instance as the python is the default entry point
docker compose exec mcp python ingest_unstructured.py
```