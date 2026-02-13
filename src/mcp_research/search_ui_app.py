from __future__ import annotations

import os
import time
from typing import Any, Dict

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response
except ImportError:  # pragma: no cover - keep importable without FastAPI
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]
    HTMLResponse = str  # type: ignore[assignment]
    Response = str  # type: ignore[assignment]

try:
    from mcp_research import mcp_app as mcp_tools
except Exception:  # pragma: no cover - optional runtime dependency chain
    mcp_tools = None  # type: ignore[assignment]


def _load_ui_html() -> str:
    try:
        from importlib.resources import files as pkg_files  # Python 3.9+

        return pkg_files("mcp_research.search_ui_static").joinpath("index.html").read_text(
            encoding="utf-8"
        )
    except Exception:  # pragma: no cover
        return "<html><body><h1>Search UI assets missing</h1></body></html>"


def _require_tools():
    if mcp_tools is None:
        raise RuntimeError("mcp_app is unavailable")
    return mcp_tools


def _safe_default_collection(tools) -> str:
    try:
        return tools._get_default_collection() or tools.QDRANT_COLLECTION
    except Exception:
        return tools.QDRANT_COLLECTION


def _payload_int(payload: Dict[str, Any], key: str, default: int, min_value: int, max_value: int) -> int:
    raw_value = payload.get(key, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = default
    if value < min_value:
        value = min_value
    if value > max_value:
        value = max_value
    return value


def _payload_text(payload: Dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key, default)
    return str(value).strip() if value is not None else default


def _payload_bool(payload: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _invoke_tool(tool: Any, **kwargs: Any) -> Any:
    """Call either a plain function tool or a FastMCP FunctionTool wrapper."""
    if callable(tool):
        return tool(**kwargs)
    wrapped_fn = getattr(tool, "fn", None)
    if callable(wrapped_fn):
        return wrapped_fn(**kwargs)
    raise TypeError(f"Unsupported tool object type: {type(tool).__name__}")


app = FastAPI(title="MCP Qdrant Search UI") if FastAPI is not None else None


if app is not None:

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_load_ui_html())

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/status")
    def api_status() -> Dict[str, Any]:
        try:
            tools = _require_tools()
            ping = _invoke_tool(tools.ping)
            default_collection = _safe_default_collection(tools)
            return {
                "ok": ping == "pong",
                "ping": ping,
                "qdrant_url": tools.QDRANT_URL,
                "default_collection": default_collection,
                "redis_configured": bool(tools.REDIS_URL),
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"MCP tools unavailable: {exc}") from exc

    @app.get("/api/collections")
    def api_collections() -> Dict[str, Any]:
        try:
            tools = _require_tools()
            payload = _invoke_tool(tools.list_collections)
            default_collection = _safe_default_collection(tools)
            return {
                "collections": payload.get("collections", []),
                "default_collection": default_collection,
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Failed to list collections: {exc}") from exc

    @app.post("/api/default-collection")
    def api_set_default_collection(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        collection = _payload_text(payload or {}, "collection")
        if not collection:
            raise HTTPException(status_code=400, detail="collection is required")
        try:
            tools = _require_tools()
            result = _invoke_tool(tools.set_default_collection, name=collection)
            return {
                "default_collection": result.get("default_collection"),
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to set default collection: {exc}") from exc

    @app.post("/api/search")
    def api_search(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        body = payload or {}
        query = _payload_text(body, "query")
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        top_k = _payload_int(body, "topK", 8, 1, 50)
        prefetch_k = _payload_int(body, "prefetchK", 60, 1, 500)
        collection = _payload_text(body, "collection") or None
        retrieval_mode = _payload_text(body, "retrievalMode", "hybrid") or "hybrid"
        include_partition = _payload_bool(body, "includePartition", False)
        include_document = _payload_bool(body, "includeDocument", False)

        try:
            tools = _require_tools()
            start = time.perf_counter()
            result = _invoke_tool(
                tools.search,
                query=query,
                top_k=top_k,
                prefetch_k=prefetch_k,
                collection=collection,
                retrieval_mode=retrieval_mode,
                include_partition=include_partition,
                include_document=include_document,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            active_collection = collection or _safe_default_collection(tools)
            active_mode = _payload_text(result, "retrieval_mode", retrieval_mode)
            return {
                "query": query,
                "collection": active_collection,
                "top_k": top_k,
                "prefetch_k": prefetch_k,
                "retrieval_mode": active_mode,
                "include_partition": bool(result.get("include_partition")),
                "include_document": bool(result.get("include_document")),
                "latency_ms": latency_ms,
                "results": result.get("results", []),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc

    @app.get("/api/fetch/{point_id}")
    def api_fetch(point_id: str, collection: str | None = None) -> Dict[str, Any]:
        clean_id = str(point_id).strip()
        clean_collection = str(collection).strip() if collection else None
        if not clean_id:
            raise HTTPException(status_code=400, detail="point_id is required")
        try:
            tools = _require_tools()
            return _invoke_tool(tools.fetch, id=clean_id, collection=clean_collection)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Fetch failed: {exc}") from exc

    @app.get("/api/chunk/{point_id}/document")
    def api_fetch_chunk_document(point_id: str, collection: str | None = None) -> Dict[str, Any]:
        clean_id = str(point_id).strip()
        clean_collection = str(collection).strip() if collection else None
        if not clean_id:
            raise HTTPException(status_code=400, detail="point_id is required")
        try:
            tools = _require_tools()
            return _invoke_tool(tools.fetch_chunk_document, id=clean_id, collection=clean_collection)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Fetch document failed: {exc}") from exc

    @app.get("/api/chunk/{point_id}/partition")
    def api_fetch_chunk_partition(point_id: str, collection: str | None = None) -> Dict[str, Any]:
        clean_id = str(point_id).strip()
        clean_collection = str(collection).strip() if collection else None
        if not clean_id:
            raise HTTPException(status_code=400, detail="point_id is required")
        try:
            tools = _require_tools()
            return _invoke_tool(tools.fetch_chunk_partition, id=clean_id, collection=clean_collection)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Fetch partition failed: {exc}") from exc

    @app.get("/api/chunk/{point_id}/bibtex")
    def api_fetch_chunk_bibtex(point_id: str, collection: str | None = None) -> Dict[str, Any]:
        clean_id = str(point_id).strip()
        clean_collection = str(collection).strip() if collection else None
        if not clean_id:
            raise HTTPException(status_code=400, detail="point_id is required")
        try:
            tools = _require_tools()
            return _invoke_tool(tools.fetch_chunk_bibtex, id=clean_id, collection=clean_collection)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Fetch BibTeX failed: {exc}") from exc


def main() -> None:
    import uvicorn  # pylint: disable=import-outside-toplevel

    if app is None:  # pragma: no cover
        raise RuntimeError("fastapi is required to run the search UI")

    host = os.getenv("SEARCH_UI_HOST", "0.0.0.0")
    port = int(os.getenv("SEARCH_UI_PORT", "8004"))
    uvicorn.run("mcp_research.search_ui_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
