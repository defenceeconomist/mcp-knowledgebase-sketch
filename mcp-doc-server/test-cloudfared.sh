ENDPOINT="https://mcp.heley.uk/mcp"

# 1) Initialize (capture headers so we can read Mcp-Session-Id if the server sets one)
HDR=$(mktemp)
BODY=$(mktemp)

curl -sS -D "$HDR" -o "$BODY" -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc":"2.0",
    "id":1,
    "method":"initialize",
    "params":{
      "protocolVersion":"2025-03-26",
      "capabilities":{},
      "clientInfo":{"name":"curl","version":"1.0"}
    }
  }'

echo "---- init response body ----"
cat "$BODY"; echo
echo "---- init response headers (session?) ----"
grep -i "^Mcp-Session-Id:" "$HDR" || true

SESSION=$(grep -i "^Mcp-Session-Id:" "$HDR" | awk "{print \$2}" | tr -d "\r")

# 2) Send notifications/initialized (no id, and usually no response body) :contentReference[oaicite:2]{index=2}
curl -sS -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  ${SESSION:+-H "Mcp-Session-Id: $SESSION"} \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' >/dev/null

# 3) List tools :contentReference[oaicite:3]{index=3}
curl -sS -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  ${SESSION:+-H "Mcp-Session-Id: $SESSION"} \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | cat

# 4) Call your 'search' tool (adjust name/args to match your server) :contentReference[oaicite:4]{index=4}
curl -sS -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  ${SESSION:+-H "Mcp-Session-Id: $SESSION"} \
  -d '{
    "jsonrpc":"2.0",
    "id":3,
    "method":"tools/call",
    "params":{
      "name":"search",
      "arguments":{"query":"biogas"}
    }
  }' | cat
    