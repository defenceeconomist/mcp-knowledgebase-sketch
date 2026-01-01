MCP Doc Server (simple)
-----------------------

Minimal MCP server that exposes two tools over FastMCP and ships with a Docker stack that includes Cloudflare Tunnel and Keycloak backed by Postgres (persistent). The MCP endpoint now supports OAuth2/OIDC bearer tokens (Keycloak) for ChatGPT Connector / MCP OAuth.

- `search(query: str)`: simple substring search across `.txt`/`.md` docs in `DOCS_DIR`
- `fetch(id: str)`: return full text for a doc id

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

Docker / Cloudflare / Keycloak (compose)
----------------------------------------

`docker-compose.yml` now starts the MCP server, Cloudflare tunnel, Keycloak, and a Postgres backing database (persistent volume `keycloak-db-data`).

Example run:

```bash
cd mcp-doc-server
export DOCS_DIR=./docs
export MCP_RESOURCE=https://<your-tunnel-host>/mcp   # or http://localhost:8000
export KEYCLOAK_ISSUER=https://<auth-host>/auth/realms/<realm>   # public issuer
export KEYCLOAK_VERIFY_TLS=1                                # 0 to skip verify (not recommended)
export REQUIRED_SCOPES=tools.read                           # comma-separated
export CF_TUNNEL_TOKEN=<cloudflare_tunnel_token>
export KEYCLOAK_DB_PASSWORD=<db_password>
export KEYCLOAK_ADMIN=admin
export KEYCLOAK_ADMIN_PASSWORD=<admin_password>
docker compose up -d
```

- MCP: `http://localhost:8000/mcp` (protected; returns 401/403 without token)
- Keycloak admin console: `http://localhost:8080/auth` (data persisted in `keycloak-db-data`)
- Configure Cloudflare Tunnel routes/ingress in the dashboard to point your hostname(s) to `mcp:8000` and optionally `keycloak:8080`.

Connecting from ChatGPT UI
--------------------------

Add a remote MCP server in ChatGPT pointing at your Cloudflare hostname + `/mcp`, e.g. `https://mcp.heley.uk/mcp`.

- Protected resource metadata: `GET /.well-known/oauth-protected-resource` advertises `resource`, `authorization_servers`, and supported scopes; the `WWW-Authenticate` header on 401/403 points to it.
- Tokens: Keycloak access tokens must have `aud` that includes `MCP_RESOURCE` and include the required scopes (in `scope` claim or `realm_access.roles`).
- If you want an open endpoint, unset `KEYCLOAK_ISSUER` and remove the middleware/auth env from compose.

Endpoints
---------

- `/health`: basic health probe
- `/mcp`: MCP JSON-RPC endpoint
- `/.well-known/oauth-protected-resource`: MCP OAuth resource metadata for the server
- Keycloak admin UI: `/auth` on port `8080` (if running the compose stack)
