# mcp-research

Local research retrieval stack built around:
- `FastMCP` for tool exposure
- `Qdrant` for hybrid vector search
- `Redis` for document/chunk metadata
- `MinIO` for object storage and event-driven ingest
- `Nginx` reverse proxy fronting resolver/UI endpoints on `http://localhost:8080`

## Quick Start

1. Configure environment values in `.env` (at minimum: `UNSTRUCTURED_API_KEY`).
2. Start the stack:

```bash
docker compose up -d --build
```

3. Ingest data:

```bash
# MinIO event-driven ingest (already part of docker-compose stack)
# Upload PDFs to watched MinIO bucket(s); minio_ingest will process and upsert automatically.
docker compose logs -f minio_ingest
```

Citation links resolve through the reverse proxy by default (`LINK_RESOLVER_MODE=proxy`), so clients stay on the `/r/*` routes instead of direct MinIO hosts.


## MCP Server

The MCP server runs from:

```bash
python mcp_cli.py mcp-app
```

Default transport is HTTP on port `8000`.

## CLI Commands

`mcp_cli.py` is the unified CLI entrypoint:

```bash
python mcp_cli.py <command> [args...]
```

Key commands:

- `python mcp_cli.py mcp-app`
- `python mcp_cli.py minio-ingest`
- `python mcp_cli.py minio-ops ...`
- `python mcp_cli.py ingest-missing-minio --dry-run`
- `python mcp_cli.py upsert-chunks --help`
- `python mcp_cli.py upload-data-to-redis --help`
- `python mcp_cli.py purge-v1-schema --help`
- `python mcp_cli.py hybrid-search --help`
- `python mcp_cli.py bibtex-autofill --help`

### MinIO Operations

These commands cover bucket/file management plus upload ingest with Unstructured:

```bash
# Add a bucket
python mcp_cli.py minio-ops add-bucket research-pdfs

# Upload a file and ingest immediately with Unstructured (default behavior)
python mcp_cli.py minio-ops upload-file research-pdfs ./data/paper.pdf --create-bucket

# Upload a file without direct ingest (rely on event-driven minio_ingest listener)
python mcp_cli.py minio-ops upload-file research-pdfs ./data/paper.pdf --no-ingest

# Remove one file from a bucket and delete ingested vectors/metadata (default behavior)
python mcp_cli.py minio-ops remove-file research-pdfs paper.pdf

# Delete a bucket (must be empty)
python mcp_cli.py minio-ops delete-bucket research-pdfs

# Force-delete a bucket and all contained objects
python mcp_cli.py minio-ops delete-bucket research-pdfs --force
```

## Documentation

Focused documentation for ingest flow, schema, and MCP tools:

- `docs/tool-documentation.md`

Core shared URL/source reference helpers are centralized in:

- `src/mcp_research/citation_utils.py`
