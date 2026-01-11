import json
import logging
import os
from typing import Any, Dict, List

from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient

from mcp_research import hybrid_search

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def load_dotenv(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_URL = os.getenv("QDRANT_URL") or f"http://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "unstructured")
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

_qdrant_client: QdrantClient | None = None
_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None
_redis_client = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def _get_models() -> tuple[TextEmbedding, SparseTextEmbedding]:
    global _dense_model, _sparse_model
    if _dense_model is None:
        _dense_model = TextEmbedding(model_name=DENSE_MODEL)
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _dense_model, _sparse_model


def _get_redis_client():
    global _redis_client
    if not REDIS_URL:
        return None
    if redis is None:
        logger.warning("REDIS_URL set but redis package is missing; skipping Redis")
        return None
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL)
    return _redis_client


def _default_collection_key() -> str:
    return f"{REDIS_PREFIX}:qdrant:default_collection"


def _decode_redis_value(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def _get_default_collection() -> str | None:
    client = _get_redis_client()
    if not client:
        return None
    return _decode_redis_value(client.get(_default_collection_key()))

# --------------------------------------------------------------------
# Auth (GitHub OAuth for ChatGPT UI)
# --------------------------------------------------------------------
auth = GitHubProvider(
    client_id=os.environ["FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID"],
    client_secret=os.environ["FASTMCP_SERVER_AUTH_GITHUB_CLIENT_SECRET"],
    base_url=os.environ["FASTMCP_SERVER_AUTH_GITHUB_BASE_URL"],
)

mcp = FastMCP(name="My GitHub-OAuth MCP Server", auth=auth)

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

@mcp.tool
def list_collections() -> Dict[str, List[str]]:
    """List Qdrant collections."""
    client = _get_qdrant_client()
    collections = client.get_collections()
    names = [entry.name for entry in collections.collections]
    return {"collections": names}


@mcp.tool
def set_default_collection(name: str) -> Dict[str, str]:
    """Set the default Qdrant collection name in Redis."""
    client = _get_qdrant_client()
    if not client.collection_exists(name):
        raise ValueError(f"Collection not found: {name}")
    redis_client = _get_redis_client()
    if not redis_client:
        raise RuntimeError("REDIS_URL is required to store the default collection")
    redis_client.set(_default_collection_key(), name)
    return {"default_collection": name}


@mcp.tool
def search(
    query: str,
    top_k: int = 5,
    prefetch_k: int = 40,
    collection: str | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search indexed chunks via hybrid (dense+sparse) retrieval.
    Returns results with ids that can be passed to fetch().
    """
    client = _get_qdrant_client()
    dense_model, sparse_model = _get_models()

    collection_name = collection or _get_default_collection() or QDRANT_COLLECTION
    response = hybrid_search.hybrid_search(
        client=client,
        collection_name=collection_name,
        dense_model=dense_model,
        sparse_model=sparse_model,
        query_text=query,
        top_k=top_k,
        prefetch_k=prefetch_k,
    )

    results: List[Dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            {
                "id": str(point.id),
                "score": point.score,
                "source": payload.get("source"),
                "pages": payload.get("pages", []),
                "text": payload.get("text", ""),
            }
        )

    return {"results": results}


@mcp.tool
def fetch(id: str, collection: str | None = None) -> Dict[str, Any]:
    """
    Fetch a single chunk by id for deep retrieval.
    """
    client = _get_qdrant_client()
    collection_name = collection or _get_default_collection() or QDRANT_COLLECTION
    points = client.retrieve(
        collection_name=collection_name,
        ids=[id],
        with_payload=True,
    )
    if not points:
        return {"id": id, "found": False}

    point = points[0]
    payload = point.payload or {}
    return {
        "id": str(point.id),
        "found": True,
        "source": payload.get("source"),
        "pages": payload.get("pages", []),
        "text": payload.get("text", ""),
    }

def main() -> None:
    # Remote deployment uses HTTP transport; default MCP path is /mcp
    mcp.run(transport="http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
