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
COPY mcp_app.py /app/mcp_app.py
COPY ingest_unstructured.py /app/ingest_unstructured.py
COPY hybrid_search.py /app/hybrid_search.py
COPY upsert_chunks.py /app/upsert_chunks.py
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Default to running the MCP app, but keep a Python entrypoint so other scripts
# (e.g., ingest_unstructured.py) can be invoked by overriding the command.
ENTRYPOINT ["python"]
CMD ["mcp_app.py"]
