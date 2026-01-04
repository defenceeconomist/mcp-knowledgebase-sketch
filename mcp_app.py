import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Auth (GitHub OAuth for ChatGPT UI)
# --------------------------------------------------------------------
auth = GitHubProvider(
    client_id=os.environ["FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID"],
    client_secret=os.environ["FASTMCP_SERVER_AUTH_GITHUB_CLIENT_SECRET"],
    base_url=os.environ["FASTMCP_SERVER_AUTH_GITHUB_BASE_URL"],
)

mcp = FastMCP(name="My GitHub-OAuth MCP Server", auth=auth)

# --------------------------------------------------------------------
# Env helpers
# --------------------------------------------------------------------
def load_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s, using %d", key, raw, default)
        return default


def load_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%s, using %.3f", key, raw, default)
        return default


def mcp_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrap a payload in the MCP *text* content structure that
    ChatGPT Deep Research expects.
    """
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = load_env_int("QDRANT_PORT", 6333)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")

# If you are using named vectors in Qdrant, set this to the name
# (e.g. "text"). Otherwise leave unset / empty.
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "").strip() or None

SEARCH_TOP_K = load_env_int("SEARCH_TOP_K", 15)
SEARCH_MAX_CHUNKS_PER_DOC = load_env_int("SEARCH_MAX_CHUNKS_PER_DOC", 3)
HYBRID_ALPHA = load_env_float("HYBRID_ALPHA", 0.7)

FETCH_NEIGHBOUR_WINDOW = load_env_int("FETCH_NEIGHBOUR_WINDOW", 1)
FETCH_MAX_CHARS = load_env_int("FETCH_MAX_CHARS", 8000)

_qdrant_client: Optional[QdrantClient] = None
_embedding_model: Optional[SentenceTransformer] = None

# --------------------------------------------------------------------
# Lazily initialised clients
# --------------------------------------------------------------------
def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            logger.info("Initialising QdrantClient via URL=%s", QDRANT_URL)
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=False,
            )
        else:
            logger.info("Initialising QdrantClient via host=%s port=%s", QDRANT_HOST, QDRANT_PORT)
            _qdrant_client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY,
                prefer_grpc=False,
            )
    return _qdrant_client


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s'", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_query(text: str) -> List[float]:
    model = get_embedding_model()
    vector = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector[0].tolist()

# --------------------------------------------------------------------
# Utility helpers for metadata & formatting
# --------------------------------------------------------------------
def format_pages(payload: Dict[str, Any]) -> Optional[str]:
    page_start = payload.get("page_start")
    page_end = payload.get("page_end")
    pages = payload.get("pages") or payload.get("page_range")

    if page_start or page_end:
        start = page_start or page_end
        end = page_end or page_start
        if start and end and start != end:
            return f"{start}-{end}"
        return f"{start or end}"

    if isinstance(pages, list):
        nums = sorted(
            {int(p) for p in pages if isinstance(p, (int, float, str)) and str(p).isdigit()}
        )
        if not nums:
            return None
        if len(nums) == 1:
            return str(nums[0])
        return f"{nums[0]}-{nums[-1]}"

    if isinstance(pages, str):
        return pages

    single_page = payload.get("page") or payload.get("page_number")
    if single_page is not None:
        return str(single_page)

    return None


def build_url(collection: str, doc_id: str, payload: Dict[str, Any], chunk_id: str) -> str:
    page_hint = payload.get("page_start") or payload.get("page_number")
    if page_hint:
        return f"internal://{collection}/{doc_id}#page={page_hint}"
    return f"internal://{collection}/{doc_id}#chunk={chunk_id}"


def build_snippet(text: str, max_len: int = 280) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    cutoff = cleaned.rfind(" ", 0, max_len)
    if cutoff <= 0:
        cutoff = max_len
    return cleaned[:cutoff].rstrip() + "..."


def get_doc_id(payload: Dict[str, Any]) -> str:
    return (
        payload.get("doc_id")
        or payload.get("source")
        or payload.get("document_id")
        or "unknown"
    )


def fuse_scores(dense_score: float, lexical_score: float, alpha: float) -> float:
    return alpha * dense_score + (1 - alpha) * lexical_score


def lexical_score(text: str, tokens: Sequence[str]) -> float:
    if not tokens or not text:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for token in tokens if token and token in lowered)
    return hits / max(len(tokens), 1)

# --------------------------------------------------------------------
# Qdrant search wrapper – simplified to avoid AttributeError
# --------------------------------------------------------------------
def _search_qdrant(
    client: QdrantClient,
    vector: List[float],
    limit: int,
) -> List[ScoredPoint]:
    """
    Use the current QdrantClient .search API.

    If you're using named vectors, we use the 'using' argument which is
    supported in recent qdrant-client versions.
    """
    kwargs: Dict[str, Any] = {
        "collection_name": QDRANT_COLLECTION,
        "query_vector": vector,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }

    if QDRANT_VECTOR_NAME:
        # Option 1: using named vector routing
        kwargs["using"] = QDRANT_VECTOR_NAME

    logger.debug("Calling Qdrant search with kwargs=%r", kwargs)

    # This call is stable across recent versions of qdrant-client
    return client.search(**kwargs)

# --------------------------------------------------------------------
# MCP tools: misc
# --------------------------------------------------------------------
@mcp.tool
async def whoami() -> dict:
    """Return info about the authenticated GitHub user."""
    token = get_access_token()
    # Be defensive in case there is some misconfiguration
    claims = getattr(token, "claims", {}) or {}
    return {
        "login": claims.get("login"),
        "name": claims.get("name"),
        "email": claims.get("email"),
    }


@mcp.tool
def ping() -> str:
    """Simple health-check."""
    return "pong"

# --------------------------------------------------------------------
# Scoring & formatting of search results for Deep Research
# --------------------------------------------------------------------
def _score_and_filter_results(
    points: List[ScoredPoint],
    query_tokens: Sequence[str],
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, float, float, ScoredPoint]] = []

    for point in points:
        payload = point.payload or {}
        dense = float(point.score or 0.0)
        lex = lexical_score(payload.get("text") or "", query_tokens)
        combined = fuse_scores(dense, lex, HYBRID_ALPHA) if query_tokens else dense
        scored.append((combined, dense, lex, point))

    # Sort by hybrid score (descending)
    scored.sort(key=lambda item: item[0], reverse=True)

    kept: List[Dict[str, Any]] = []
    per_doc: Dict[str, int] = {}

    for combined, dense, lex, point in scored:
        payload = point.payload or {}
        doc_id = get_doc_id(payload)

        count = per_doc.get(doc_id, 0)
        if count >= SEARCH_MAX_CHUNKS_PER_DOC:
            continue

        per_doc[doc_id] = count + 1

        chunk_id = payload.get("chunk_id") or str(point.id)
        title = payload.get("title") or doc_id
        page_str = format_pages(payload)
        url = build_url(QDRANT_COLLECTION, doc_id, payload, chunk_id)
        snippet = build_snippet(payload.get("text") or "")

        metadata = {
            "doc_id": doc_id,
            "collection": QDRANT_COLLECTION,
            "pages": page_str,
            "chunk_index": payload.get("chunk_index"),
            "source": payload.get("source"),
        }

        kept.append(
            {
                "id": f"{QDRANT_COLLECTION}:{doc_id}:{chunk_id}",
                "title": title,
                "url": url,
                "snippet": snippet,
                "score": round(combined, 4),
                "metadata": metadata,
            }
        )

        if len(kept) >= SEARCH_TOP_K:
            break

    return kept

# --------------------------------------------------------------------
# MCP tool: search
# --------------------------------------------------------------------
@mcp.tool
async def search(query: str) -> Dict[str, Any]:
    """
    Hybrid search over Qdrant with dense + lexical fusion.

    ChatGPT Deep Research expects:
    {
      "content": [{
        "type": "text",
        "text": "{\"results\": [...]}"
      }]
    }
    """
    trimmed = (query or "").strip()
    if not trimmed:
        return mcp_response(
            {"error": "EMPTY_QUERY", "message": "Query must not be empty."}
        )

    logger.info("search query='%s'", trimmed[:120])

    try:
        vector = embed_query(trimmed)
        client = get_qdrant_client()

        dense_limit = max(SEARCH_TOP_K * 2, SEARCH_TOP_K)
        dense_results = _search_qdrant(client, vector, dense_limit)

        tokens = [t.lower() for t in trimmed.split() if t.strip()]
        results = _score_and_filter_results(dense_results, tokens)

        logger.info("search returned ids=%s", [r["id"] for r in results])

        return mcp_response({"results": results})

    except Exception as exc:
        logger.exception("Search failed: %s", exc)
        return mcp_response(
            {
                "error": "INTERNAL_ERROR",
                "message": "Search failed.",
                "details": {"type": type(exc).__name__, "str": str(exc)},
            }
        )

# --------------------------------------------------------------------
# Helpers for fetch
# --------------------------------------------------------------------
def _collect_neighbours(
    client: QdrantClient,
    collection: str,
    payload: Dict[str, Any],
    window: int,
) -> List[Dict[str, Any]]:
    if window <= 0:
        return []

    chunk_index = payload.get("chunk_index")
    doc_key = (
        "doc_id"
        if "doc_id" in payload
        else "source"
        if "source" in payload
        else None
    )

    if chunk_index is None or doc_key is None:
        return []

    neighbours: List[Dict[str, Any]] = []
    doc_val = payload.get(doc_key)

    for offset in range(1, window + 1):
        for neighbour_idx in (chunk_index - offset, chunk_index + offset):
            if neighbour_idx < 0:
                continue

            filt = Filter(
                must=[
                    FieldCondition(key=doc_key, match=MatchValue(value=doc_val)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=neighbour_idx)),
                ]
            )

            page, _ = client.scroll(
                collection_name=collection,
                filter=filt,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            if page:
                neighbours.append(page[0].payload or {})

    return neighbours


def _merge_pages(payloads: List[Dict[str, Any]]) -> Optional[str]:
    pages: List[str] = []

    for payload in payloads:
        page_str = format_pages(payload)
        if page_str:
            pages.append(page_str)

    if not pages:
        return None

    if len(pages) == 1:
        return pages[0]

    nums: List[int] = []
    for item in pages:
        try:
            nums.append(int(str(item).split("-")[0]))
        except (ValueError, AttributeError):
            # Fall back to joining raw page strings
            return ", ".join(sorted(set(pages)))

    return f"{min(nums)}-{max(nums)}"

# --------------------------------------------------------------------
# MCP tool: fetch
# --------------------------------------------------------------------
@mcp.tool
async def fetch(id: str) -> Dict[str, Any]:
    """
    Fetch a chunk (and optional neighbours) by composite id.

    Expected id format for Deep Research:
        "<collection>:<doc_id>:<chunk_id>"
    """
    if not id or ":" not in id:
        return mcp_response(
            {
                "error": "INVALID_ID",
                "message": "Expected id format '<collection>:<doc_id>:<chunk_id>'.",
            }
        )

    parts = id.split(":")
    if len(parts) != 3:
        return mcp_response(
            {
                "error": "INVALID_ID",
                "message": "Expected id format '<collection>:<doc_id>:<chunk_id>'.",
            }
        )

    collection_name, doc_id, chunk_id = parts
    logger.info("fetch id=%s", id)

    try:
        client = get_qdrant_client()
        point = None

        # Preferred path: retrieve by point id
        try:
            record = client.retrieve(
                collection_name=collection_name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=False,
            )
            point = record[0] if record else None
        except TypeError:
            # Older clients: get_points
            resp = client.get_points(
                collection_name=collection_name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=False,
            )
            point = resp[0] if resp else None

        if not point:
            # Fallback: look up by payload.chunk_id if you stored it there
            filt = Filter(
                must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]
            )
            page, _ = client.scroll(
                collection_name=collection_name,
                filter=filt,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            point = page[0] if page else None

        if not point:
            return mcp_response(
                {
                    "error": "DOCUMENT_NOT_FOUND",
                    "message": "No chunk found for the given id.",
                }
            )

        payload = point.payload or {}
        main_text = (payload.get("text") or "").strip()

        neighbours = _collect_neighbours(
            client, collection_name, payload, FETCH_NEIGHBOUR_WINDOW
        )

        texts = [main_text] + [
            (n.get("text") or "") for n in neighbours if n.get("text")
        ]
        combined_text = " ".join(t for t in texts if t).strip()

        if len(combined_text) > FETCH_MAX_CHARS:
            combined_text = combined_text[:FETCH_MAX_CHARS]

        all_payloads = [payload] + neighbours
        pages = _merge_pages(all_payloads)

        title = payload.get("title") or doc_id
        url = build_url(collection_name, doc_id, payload, chunk_id)

        metadata: Dict[str, Any] = {
            "doc_id": doc_id,
            "collection": collection_name,
            "pages": pages,
            "chunk_index": payload.get("chunk_index"),
            "total_chunks": payload.get("total_chunks"),
            "source": payload.get("source"),
        }

        # Pass through some optional / nice-to-have fields
        for optional in ("mime_type", "section_title"):
            if optional in payload:
                metadata[optional] = payload[optional]

        response = {
            "id": id,
            "title": title,
            "text": combined_text,
            "url": url,
            "metadata": metadata,
        }

        return mcp_response(response)

    except Exception as exc:
        logger.exception("Fetch failed: %s", exc)
        return mcp_response(
            {
                "error": "INTERNAL_ERROR",
                "message": "Fetch failed.",
                "details": {"type": type(exc).__name__, "str": str(exc)},
            }
        )

# --------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Remote deployment uses HTTP transport; default MCP path is /mcp
    mcp.run(transport="http", host="0.0.0.0", port=8000)
