# mcp-research

Local research retrieval stack built around:
- `FastMCP` for tool exposure
- `Qdrant` for hybrid vector search
- `Redis` for document/chunk metadata
- `MinIO` for object storage and event-driven ingest

## Quick Start

1. Configure environment values in `.env` (at minimum: `UNSTRUCTURED_API_KEY`).
2. Start the stack:

```bash
docker compose up -d --build
```

3. Ingest data:

```bash
# Local PDFs -> Unstructured -> Redis (optional disk output)
python mcp_cli.py ingest-unstructured

# Chunk payloads -> Qdrant
python mcp_cli.py upsert-chunks --source redis --redis-url redis://localhost:6379/0
```

For MinIO-driven ingest, run the `minio_ingest` service (already included in `docker-compose.yml`).

## MCP Server

The MCP server runs from:

```bash
python mcp_cli.py mcp-app
```

Default transport is HTTP on port `8000`.

## Documentation

Focused documentation for ingest flow, schema, and MCP tools:

- `docs/tool-documentation.md`
