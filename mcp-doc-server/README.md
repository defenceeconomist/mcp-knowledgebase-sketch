MCP Doc Server (simple)
-----------------------

Minimal MCP server that exposes two tools over FastMCP:

- `search(query: str)`: simple substring search across `.txt`/`.md` docs in `DOCS_DIR`
- `fetch(id: str)`: return full text for a doc id

No authentication; meant to be fronted by a Cloudflare tunnel if you want to put it on the internet.

Quick start (Python)
--------------------

```bash
cd mcp-doc-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DOCS_DIR=./docs
export MCP_RESOURCE=https://mcp.heley.uk      # Public origin (no /mcp suffix), used for doc URLs; default http://localhost:8000
python server.py                              # or: uvicorn server:app --host 0.0.0.0 --port 8000
```

Docker / Cloudflared
--------------------

- `docker build -t mcp-doc-server .`
- Run with `DOCS_DIR=/docs` and mount your docs read-only: `-v ./docs:/docs:ro`
- `docker-compose.yml` includes:
  - `mcp` service publishing `:8000`
  - `cloudflared` tunnel to expose the service publicly; set `CF_TUNNEL_TOKEN` in your environment

Connecting from ChatGPT UI
--------------------------

When adding a remote MCP server in ChatGPT, point to your tunnel URL, e.g. `https://mcp.heley.uk/mcp`. No OAuth is required.

Endpoints
---------

- `/health`: basic health probe
- `/mcp`: MCP JSON-RPC endpoint
