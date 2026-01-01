import logging
import os
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import httpx
import jwt
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from jwt import InvalidTokenError, PyJWKClient

# ===== Configuration =====
MCP_RESOURCE = os.environ.get("MCP_RESOURCE", "http://localhost:8000").rstrip("/")
DOC_RESOURCE_BASE = MCP_RESOURCE.rsplit("/mcp", 1)[0] or MCP_RESOURCE
KEYCLOAK_ISSUER = os.environ.get("KEYCLOAK_ISSUER", "").rstrip("/")
VERIFY_TLS = os.environ.get("KEYCLOAK_VERIFY_TLS", "1").lower() not in {"0", "false", "no"}
REQUIRED_SCOPES: List[str] = [
    scope.strip() for scope in os.environ.get("REQUIRED_SCOPES", "tools.read").split(",") if scope.strip()
] or ["tools.read"]
REQUIRED_SCOPES_STR = " ".join(REQUIRED_SCOPES)
RESOURCE_METADATA_URL = f"{MCP_RESOURCE}/.well-known/oauth-protected-resource"

# Local docs directory
DOCS_DIR = Path(os.environ.get("DOCS_DIR", "./docs")).resolve()

# Caches for discovery and JWKS
_discovery_cache: Dict[str, object] = {"expires": 0.0, "data": None}
_jwk_clients: Dict[str, PyJWKClient] = {}

logger = logging.getLogger("mcp-auth")
logging.basicConfig(level=logging.INFO)

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
                    "url": f"{DOC_RESOURCE_BASE}/doc/{quote(docid)}",
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


# ===== Auth helpers =====
def _build_auth_header(error: str | None = None) -> Dict[str, str]:
    parts = [f'resource_metadata="{RESOURCE_METADATA_URL}"', f'scope="{REQUIRED_SCOPES_STR}"']
    if error:
        parts.insert(0, f'error="{error}"')
    return {"WWW-Authenticate": "Bearer " + ", ".join(parts)}


async def _get_discovery() -> Dict:
    if not KEYCLOAK_ISSUER:
        raise HTTPException(status_code=500, detail="KEYCLOAK_ISSUER not configured")
    now = time.time()
    if _discovery_cache["data"] and now < _discovery_cache["expires"]:
        return _discovery_cache["data"]  # type: ignore[return-value]

    url = f"{KEYCLOAK_ISSUER}/.well-known/openid-configuration"
    try:
        async with httpx.AsyncClient(verify=VERIFY_TLS, timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("OIDC discovery failed: %s", exc)
        raise HTTPException(status_code=401, detail="invalid_token", headers=_build_auth_header("invalid_token"))

    _discovery_cache["data"] = data
    _discovery_cache["expires"] = now + 300
    return data


def _get_jwk_client(jwks_uri: str) -> PyJWKClient:
    client = _jwk_clients.get(jwks_uri)
    if client is None:
        # Use a tiny subclass to honor VERIFY_TLS while staying compatible with common PyJWT versions.
        class TLSAwarePyJWKClient(PyJWKClient):  # type: ignore
            def fetch_data(self_inner):
                resp = requests.get(self_inner.jwks_uri, timeout=5.0, verify=VERIFY_TLS)
                resp.raise_for_status()
                return resp.json()

        client = TLSAwarePyJWKClient(jwks_uri, cache_keys=True)
        _jwk_clients[jwks_uri] = client
    return client


def _has_required_scopes(payload: Dict) -> bool:
    scope_claims = set(str(payload.get("scope", "")).split())
    realm_roles = set(payload.get("realm_access", {}).get("roles", []))
    required = set(REQUIRED_SCOPES)
    return required.issubset(scope_claims) or required.issubset(realm_roles)


async def _verify_token(token: str) -> Dict:
    discovery = await _get_discovery()
    issuer = str(discovery.get("issuer", "")).rstrip("/")
    expected_issuer = KEYCLOAK_ISSUER
    jwks_uri = discovery.get("jwks_uri")
    if not issuer or not jwks_uri:
        logger.warning("Discovery missing issuer or jwks_uri")
        raise HTTPException(status_code=401, detail="invalid_token", headers=_build_auth_header("invalid_token"))
    if expected_issuer and issuer != expected_issuer:
        logger.warning("Issuer mismatch: discovery '%s' vs expected '%s'", issuer, expected_issuer)
        raise HTTPException(status_code=401, detail="invalid_token", headers=_build_auth_header("invalid_token"))

    try:
        jwk_client = _get_jwk_client(jwks_uri)
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        valid_audiences = {aud for aud in [MCP_RESOURCE, MCP_RESOURCE.rstrip("/"), DOC_RESOURCE_BASE] if aud}
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=list(valid_audiences),
            issuer=issuer,
            options={"require": ["exp", "iss"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token_expired", headers=_build_auth_header("invalid_token"))
    except jwt.InvalidAudienceError:
        logger.warning("Audience mismatch for resource '%s'", MCP_RESOURCE)
        raise HTTPException(status_code=401, detail="invalid_audience", headers=_build_auth_header("invalid_token"))
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="invalid_token", headers=_build_auth_header("invalid_token"))
    except Exception as exc:
        logger.warning("Token validation error: %s", exc)
        raise HTTPException(status_code=401, detail="invalid_token", headers=_build_auth_header("invalid_token"))

    token_issuer = str(payload.get("iss", "")).rstrip("/")
    if expected_issuer and token_issuer != expected_issuer:
        logger.warning("Issuer claim mismatch: '%s' vs expected '%s'", token_issuer, expected_issuer)
        raise HTTPException(status_code=401, detail="invalid_token", headers=_build_auth_header("invalid_token"))

    if not _has_required_scopes(payload):
        return _forbid_insufficient_scope()

    return payload


def _forbid_insufficient_scope():
    header = {
        "WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{REQUIRED_SCOPES_STR}", resource_metadata="{RESOURCE_METADATA_URL}"'
    }
    raise HTTPException(status_code=403, detail="insufficient_scope", headers=header)


# ===== FastAPI wrapper =====
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(lifespan=mcp_app.lifespan, routes=[*mcp_app.routes])


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path in {"/health", "/.well-known/oauth-protected-resource"} or not path.startswith("/mcp"):
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return JSONResponse(
            {"detail": "missing_token"},
            status_code=401,
            headers=_build_auth_header(),
        )

    token = auth_header.split(" ", 1)[1].strip()
    try:
        await _verify_token(token)
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    return await call_next(request)


@app.get("/.well-known/oauth-protected-resource")
def oauth_protected_resource():
    if not KEYCLOAK_ISSUER:
        raise HTTPException(status_code=500, detail="KEYCLOAK_ISSUER not configured")
    return {
        "resource": MCP_RESOURCE,
        "authorization_servers": [KEYCLOAK_ISSUER],
        "scopes_supported": REQUIRED_SCOPES,
    }


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
