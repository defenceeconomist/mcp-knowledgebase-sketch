import argparse
import asyncio

from fastmcp import Client


async def run() -> None:
    parser = argparse.ArgumentParser(
        description="Call the MCP ping tool over HTTP.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/mcp",
        help="MCP server URL (default: http://localhost:8000/mcp)",
    )
    parser.add_argument(
        "--auth",
        default="oauth",
        choices=["oauth", "none", "token"],
        help="Auth mode: oauth (default), none, or token",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token (used when --auth token).",
    )
    args = parser.parse_args()

    if args.auth == "token" and not args.token:
        raise SystemExit("--token is required when --auth token")

    auth = None
    if args.auth == "oauth":
        auth = "oauth"
    elif args.auth == "token":
        auth = args.token

    try:
        async with Client(args.url, auth=auth) as client:
            result = await client.call_tool("ping", {})
            if getattr(result, "content", None):
                print(result.content[0].text)
            else:
                print(result)
    except RuntimeError as exc:
        raise SystemExit(
            "Ping failed; ensure the OAuth browser flow completes and the callback URL is reachable."
        ) from exc


if __name__ == "__main__":
    asyncio.run(run())
