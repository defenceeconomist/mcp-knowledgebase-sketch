import os
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI
from fastmcp import FastMCP

# Public origin of the service (used for doc URLs). No trailing slash.
RESOURCE = os.environ.get("MCP_RESOURCE", "http://localhost:8000").rstrip("/")

# Local docs directory
DOCS_DIR = Path(os.environ.get("DOCS_DIR", "./docs")).resolve()

# ===== FastMCP tools =====
mcp = FastMCP("Docs MCP")


def _iter_docs():
    if not DOCS_DIR.exists():
        return
    for p in DOCS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            yield p


def _doc_id(p: Path) -> str:
    return str(p.relative_to(DOCS_DIR)).replace("\\", "/")


@mcp.tool
def search(query: str):
    """Return doc ids whose content contains the query (simple demo search)."""
    q = query.lower().strip()
    results = []
    for p in _iter_docs():
        txt = p.read_text(encoding="utf-8", errors="ignore").lower()
        if q and q in txt:
            docid = _doc_id(p)
            results.append(
                {
                    "id": docid,
                    "title": p.stem,
                    "url": f"{RESOURCE.rsplit('/mcp', 1)[0]}/doc/{quote(docid)}",
                }
            )
    return {"results": results[:10]}


@mcp.tool
def fetch(id: str):
    """Return the full text of a doc by id."""
    target = (DOCS_DIR / id).resolve()
    if not str(target).startswith(str(DOCS_DIR)) or not target.exists() or not target.is_file():
        return {"id": id, "title": id, "text": "[Document not found]"}
    return {"id": id, "title": target.stem, "text": target.read_text(encoding="utf-8", errors="ignore")}


# ===== FastAPI wrapper =====
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(lifespan=mcp_app.lifespan, routes=[*mcp_app.routes])


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
