import os
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token

auth = GitHubProvider(
    client_id=os.environ["FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID"],
    client_secret=os.environ["FASTMCP_SERVER_AUTH_GITHUB_CLIENT_SECRET"],
    # Public URL where the OAuth endpoints are reachable (your tunnel hostname)
    base_url=os.environ["FASTMCP_SERVER_AUTH_GITHUB_BASE_URL"],  # e.g. https://mcp.yourdomain.com
    # redirect_path defaults to /auth/callback
)

mcp = FastMCP(name="My GitHub-OAuth MCP Server", auth=auth)

@mcp.tool
async def whoami() -> dict:
    """Return info about the authenticated GitHub user."""
    token = get_access_token()
    return {
        "login": token.claims.get("login"),
        "name": token.claims.get("name"),
        "email": token.claims.get("email"),
    }

@mcp.tool
def ping() -> str:
    return "pong"

if __name__ == "__main__":
    # Remote deployment uses HTTP transport; default MCP path is /mcp
    mcp.run(transport="http", host="0.0.0.0", port=8000)