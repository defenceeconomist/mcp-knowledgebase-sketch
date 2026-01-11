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
- `FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID`
- `FASTMCP_SERVER_AUTH_GITHUB_CLIENT_SECRET`
- `FASTMCP_SERVER_AUTH_GITHUB_BASE_URL`

(Optional) add:
- `UNSTRUCTURED_API_KEY`
- `QDRANT_URL` (default `http://localhost:6333`)
- `QDRANT_COLLECTION` (default `pdf_chunks`)

## 3) Run the MCP server
To test OAuth locally, expose your server with a public URL (Cloudflare Tunnel, ngrok, etc.). Set `FASTMCP_SERVER_AUTH_GITHUB_BASE_URL` to the host root (no `/mcp` suffix) and update your GitHub OAuth callback URL to `<base_url>/auth/callback`.

```bash
python mcp_app.py
```
Server listens on `http://0.0.0.0:8000`.

Test the ping tool from your terminal (uses OAuth login in a browser):
```bash
python examples/ping_mcp.py --url https://mcp.heley.uk/mcp
```

## 4) Run Qdrant (optional)
```bash
docker compose up qdrant
```

## 5) Hybrid search demo
```bash
python hybrid_search.py "hybrid search in qdrant" --recreate
```

## 6) Unstructured ingest (optional)
```bash
python ingest_unstructured.py
python upsert_chunks.py --chunks-dir ./data/chunks --collection test_collection
```
