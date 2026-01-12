import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse

from mcp_research.link_resolver import resolve_link


app = FastAPI(title="Citation Link Resolver", version="0.1.0")


@app.get("/r/doc")
def resolve_doc(
    ref: Optional[str] = Query(default=None, description="doc:// reference"),
    bucket: Optional[str] = Query(default=None),
    key: Optional[str] = Query(default=None),
    version_id: Optional[str] = Query(default=None),
    page: Optional[int] = Query(default=None),
    page_start: Optional[int] = Query(default=None),
    page_end: Optional[int] = Query(default=None),
    mode: Optional[str] = Query(default=None),
):
    """Resolve a citation reference to a target URL."""
    try:
        result = resolve_link(
            source_ref=ref,
            bucket=bucket,
            key=key,
            version_id=version_id,
            page=page,
            page_start=page_start,
            page_end=page_end,
            mode=mode,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=result["url"], status_code=302)


@app.get("/r/doc.json")
def resolve_doc_json(
    ref: Optional[str] = Query(default=None, description="doc:// reference"),
    bucket: Optional[str] = Query(default=None),
    key: Optional[str] = Query(default=None),
    version_id: Optional[str] = Query(default=None),
    page: Optional[int] = Query(default=None),
    page_start: Optional[int] = Query(default=None),
    page_end: Optional[int] = Query(default=None),
    mode: Optional[str] = Query(default=None),
):
    """Resolve a citation reference and return JSON instead of redirecting."""
    try:
        result = resolve_link(
            source_ref=ref,
            bucket=bucket,
            key=key,
            version_id=version_id,
            page=page,
            page_start=page_start,
            page_end=page_end,
            mode=mode,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=result)


def main() -> None:
    host = os.getenv("RESOLVER_HOST", "0.0.0.0")
    port = int(os.getenv("RESOLVER_PORT", "8080"))
    import uvicorn

    uvicorn.run("mcp_research.resolver_app:app", host=host, port=port)


if __name__ == "__main__":
    main()
