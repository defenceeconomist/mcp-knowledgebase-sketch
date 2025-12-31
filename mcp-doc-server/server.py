import os
import json
import re
from pathlib import Path
from urllib.parse import quote

from fastmcp import FastMCP

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

mcp = FastMCP("Local Docs (search/fetch)")

DOCS_DIR = Path(os.environ.get("DOCS_DIR", "./docs")).resolve()
# Used for citations in Deep Research results. Set this to your Cloudflare HTTPS URL later.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://example.invalid").rstrip("/")

# Very small safety limits for a demo server
MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "200000"))
ALLOWED_EXTS = {".txt", ".md", ".pdf"}


def _iter_docs():
    if not DOCS_DIR.exists():
        return
    for p in DOCS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            yield p


def _read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        if PdfReader is None:
            return "[PDF support not installed: install pypdf]"
        try:
            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            return "\n".join(parts)
        except Exception as e:
            return f"[Failed to read PDF: {e}]"

    return ""


def _doc_id(path: Path) -> str:
    # stable, relative ID
    return str(path.relative_to(DOCS_DIR)).replace("\\", "/")


def _score(query: str, text: str) -> int:
    # naive keyword scoring; good enough to prove the end-to-end flow
    terms = [t for t in re.split(r"\s+", query.lower().strip()) if t]
    hay = text.lower()
    return sum(hay.count(t) for t in terms)


@mcp.tool
def search(query: str):
    """
    Return a list of relevant documents for Deep Research.

    Must return: {"content":[{"type":"text","text":"{...json...}"}]}
    """
    results = []
    for p in _iter_docs():
        doc_text = _read_text(p)
        s = _score(query, doc_text)
        if s <= 0:
            continue

        docid = _doc_id(p)
        title = p.stem
        url = f"{PUBLIC_BASE_URL}/doc/{quote(docid)}"

        results.append({"id": docid, "title": title, "url": url, "_score": s})

    # rank & trim
    results.sort(key=lambda r: r["_score"], reverse=True)
    for r in results:
        r.pop("_score", None)

    payload = {"results": results[:10]}
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False),
            }
        ]
    }


@mcp.tool
def fetch(id: str):
    """
    Return full document text for Deep Research.

    Must return: {"content":[{"type":"text","text":"{...json...}"}]}
    """
    target = (DOCS_DIR / id).resolve()
    if not str(target).startswith(str(DOCS_DIR)) or not target.exists() or not target.is_file():
        doc = {
            "id": id,
            "title": id,
            "text": "[Document not found]",
            "url": f"{PUBLIC_BASE_URL}/doc/{quote(id)}",
            "metadata": {"source": "local_files", "error": "not_found"},
        }
    else:
        text = _read_text(target)
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS] + "\n\n[Truncated]"
        doc = {
            "id": id,
            "title": target.stem,
            "text": text,
            "url": f"{PUBLIC_BASE_URL}/doc/{quote(id)}",
            "metadata": {
                "source": "local_files",
                "path": _doc_id(target),
                "content_type": target.suffix.lower().lstrip("."),
            },
        }

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(doc, ensure_ascii=False),
            }
        ]
    }


if __name__ == "__main__":
    # FastMCP remote HTTP transport (ChatGPT supports SSE + streaming HTTP).
    # The ChatGPT connector URL typically ends with /mcp/ .
    mcp.run(transport="http", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
