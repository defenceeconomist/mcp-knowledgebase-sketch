# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# (Optional but handy) system certs + curl for debugging/healthchecks
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY mcp_cli.py /app/mcp_cli.py
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Default to running the MCP app, but keep a Python entrypoint so other module
# commands can be invoked by overriding the command.
ENTRYPOINT ["python"]
CMD ["mcp_cli.py", "mcp-app"]
