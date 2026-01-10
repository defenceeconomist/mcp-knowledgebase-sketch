import json
import logging
import os

from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token


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
# Entrypoint
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Remote deployment uses HTTP transport; default MCP path is /mcp
    mcp.run(transport="http", host="0.0.0.0", port=8000)
