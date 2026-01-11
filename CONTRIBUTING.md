# Contributing

Thanks for helping improve this project. This guide covers common workflows for updating the app, documentation, and Docker images.

## Prerequisites
- Python 3.12+
- Docker + Docker Compose (optional)

## Setup
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run locally
```bash
python mcp_app.py
```

## Update the MCP app
1) Edit code in `src/mcp_research/`.
2) Restart the local server to load changes.
3) If running via Docker, rebuild the image (see below).

## Add or update tools
- Add tools in `src/mcp_research/mcp_app.py` using `@mcp.tool`.
- Keep tool inputs/outputs JSON-serializable.
- Restart the server and reconnect your MCP client UI to refresh the tool list.

## Tests
Unit tests live in `tests/`:
```bash
python -m unittest
```

## Documentation
Sphinx docs live in `docs/`:
```bash
pip install -e ".[docs]"
cd docs
make html
```
The HTML output is in `docs/build/html/`.

## Docker
Build and run the server:
```bash
docker compose up --build
```

If you change code or dependencies, rebuild the image so the container picks up changes.

## Release checklist
- Run unit tests
- Build Sphinx docs
- Update README/QUICKSTART if needed
- Rebuild Docker image
